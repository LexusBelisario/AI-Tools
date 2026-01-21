from fastapi import APIRouter, Form, Header, HTTPException
from sqlalchemy import text
import os
import json
from datetime import datetime
import re
import io
import joblib

from common_db_runtime import (
    resolve_common_context_from_token,
    set_request_context,
    clear_request_context,
    get_request_session,
)

router = APIRouter()

EXPORT_DIR = os.path.join(os.getcwd(), "exported_models")


def _extract_bearer_token(authorization: str) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authorization must be: Bearer <token>")

    return parts[1].strip()


def _validate_ident(name: str, kind: str) -> str:
    if not name:
        raise HTTPException(status_code=400, detail=f"Missing {kind}")
    if not re.match(r"^[A-Za-z0-9_]+$", name):
        raise HTTPException(status_code=400, detail=f"Invalid {kind}: {name}")
    return name


def _safe_model_path(model_path: str) -> str:
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path is required")

    norm = os.path.normpath(model_path)
    export_norm = os.path.normpath(EXPORT_DIR)

    if not norm.startswith(export_norm):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model path (must be inside {EXPORT_DIR})",
        )

    if not os.path.exists(norm):
        raise HTTPException(status_code=404, detail=f"Model file not found: {norm}")

    if not norm.lower().endswith(".pkl"):
        raise HTTPException(status_code=400, detail="model_path must be a .pkl file")

    return norm


def _parse_version_from_filename(filename: str) -> int:
    if not filename:
        return 0
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"(?:^|[_-])v(\d+)$", base, re.IGNORECASE)
    if not m:
        m = re.search(r"v(\d+)", base, re.IGNORECASE)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def _ensure_models_table(db, schema: str):
    schema = _validate_ident(schema, "schema")

    db.execute(text(f'''
        CREATE TABLE IF NOT EXISTS "{schema}"."ai_trained_models" (
            id BIGSERIAL PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            model_version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            dependent_var TEXT NULL,
            features JSONB NULL,
            metrics JSONB NULL,
            model_blob BYTEA NOT NULL,
            meta JSONB NULL
        )
    '''))

    db.execute(text(f'''
        ALTER TABLE "{schema}"."ai_trained_models"
            ADD COLUMN IF NOT EXISTS model_version INTEGER
    '''))
    db.execute(text(f'''
        ALTER TABLE "{schema}"."ai_trained_models"
            ADD COLUMN IF NOT EXISTS metrics JSONB
    '''))

    db.execute(text(f'''
        UPDATE "{schema}"."ai_trained_models"
        SET model_version = 1
        WHERE model_version IS NULL
    '''))
    db.execute(text(f'''
        ALTER TABLE "{schema}"."ai_trained_models"
        ALTER COLUMN model_version SET DEFAULT 1
    '''))
    db.execute(text(f'''
        ALTER TABLE "{schema}"."ai_trained_models"
        ALTER COLUMN model_version SET NOT NULL
    '''))
    db.commit()


def _next_model_version(db, schema: str, model_type: str) -> int:
    schema = _validate_ident(schema, "schema")
    row = db.execute(
        text(f'''
            SELECT COALESCE(MAX(model_version), 0)
            FROM "{schema}"."ai_trained_models"
            WHERE model_type = :model_type
        '''),
        {"model_type": model_type},
    ).fetchone()

    maxv = int(row[0]) if row and row[0] is not None else 0
    return maxv + 1


@router.post("/save-trained-model-local")
async def save_trained_model_local(
    model_path: str = Form(...),
    model_type: str = Form(...),
    dependent_var: str = Form(""),
    features_json: str = Form(""),
    metrics_json: str = Form(""),
    model_version: int = Form(0),
    authorization: str = Header(default=""),
    x_target_schema: str = Header(default="", alias="X-Target-Schema"),
    x_target_db: str = Header(default="", alias="X-Target-DB"),
):
    token = _extract_bearer_token(authorization)

    ctx = resolve_common_context_from_token(
        token,
        db_override=(x_target_db or None),
        schema_override=(x_target_schema or None),
    )

    set_request_context(ctx)
    db = None

    try:
        schema = _validate_ident(ctx["schema"], "schema")
        safe_path = _safe_model_path(model_path)

        with open(safe_path, "rb") as f:
            blob = f.read()

        try:
            features = json.loads(features_json) if features_json else None
        except Exception:
            features = None

        try:
            metrics = json.loads(metrics_json) if metrics_json else None
        except Exception:
            metrics = None

        meta = {
            "saved_from": "auto_after_training",
            "source_file": os.path.basename(safe_path),
            "saved_at": datetime.utcnow().isoformat(),
        }

        db = get_request_session()
        _ensure_models_table(db, schema)

        if not model_version or int(model_version) <= 0:
            guessed = _parse_version_from_filename(safe_path)
            if guessed > 0:
                model_version = guessed
            else:
                model_version = _next_model_version(db, schema, model_type)

        row = db.execute(
            text(f'''
                INSERT INTO "{schema}"."ai_trained_models"
                    (model_name, model_type, model_version, dependent_var, features, metrics, model_blob, meta)
                VALUES
                    (:model_name, :model_type, :model_version, :dependent_var,
                     CAST(:features AS JSONB), CAST(:metrics AS JSONB), :model_blob, CAST(:meta AS JSONB))
                RETURNING id
            '''),
            {
                "model_name": os.path.splitext(os.path.basename(safe_path))[0],
                "model_type": model_type,
                "model_version": int(model_version),
                "dependent_var": dependent_var or None,
                "features": json.dumps(features) if features is not None else None,
                "metrics": json.dumps(metrics) if metrics is not None else None,
                "model_blob": blob,
                "meta": json.dumps(meta),
            },
        ).fetchone()

        new_id = int(row[0]) if row else None
        db.commit()

        return {
            "success": True,
            "message": f"Saved model to Common DB table {schema}.ai_trained_models",
            "schema": schema,
            "id": new_id,
            "model_version": int(model_version),
        }

    finally:
        if db is not None:
            db.close()
        clear_request_context()


@router.get("/list-models")
async def list_models(
    authorization: str = Header(default=""),
    x_target_schema: str = Header(default="", alias="X-Target-Schema"),
    x_target_db: str = Header(default="", alias="X-Target-DB"),
):
    """
    List available models from Common Database ai_trained_models table.
    Returns model details including ID for fetching.
    """
    token = _extract_bearer_token(authorization)

    ctx = resolve_common_context_from_token(
        token,
        db_override=(x_target_db or None),
        schema_override=(x_target_schema or None),
    )

    set_request_context(ctx)
    db = None

    try:
        schema = _validate_ident(ctx["schema"], "schema")
        db = get_request_session()

        # Check if table exists
        exists = db.execute(
            text('''
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = :schema AND table_name = 'ai_trained_models'
            '''),
            {"schema": schema},
        ).fetchone()

        if not exists:
            return {"models": []}

        # Fetch model details
        rows = db.execute(text(f'''
            SELECT 
                id,
                model_name,
                model_type,
                model_version,
                dependent_var,
                features,
                created_at
            FROM "{schema}"."ai_trained_models"
            ORDER BY created_at DESC
            LIMIT 200
        ''')).fetchall()

        models = []
        for row in rows:
            model_id, name, mtype, version, dep_var, features_json, created = row
            
            # Parse features if available
            features = None
            if features_json:
                try:
                    features = json.loads(features_json) if isinstance(features_json, str) else features_json
                except:
                    features = features_json

            models.append({
                "id": model_id,
                "name": name,
                "type": mtype,
                "version": version,
                "dependent_var": dep_var,
                "features": features,
                "created_at": created.isoformat() if created else None,
                "display_name": f"{name} (v{version}) - {mtype.upper()}"
            })

        return {"models": models}

    finally:
        if db is not None:
            db.close()
        clear_request_context()


@router.get("/get-model-blob/{model_id}")
async def get_model_blob(
    model_id: int,
    authorization: str = Header(default=""),
    x_target_schema: str = Header(default="", alias="X-Target-Schema"),
    x_target_db: str = Header(default="", alias="X-Target-DB"),
):
    """
    Fetch model BLOB from Common Database by ID.
    Returns the pickled model bundle in memory.
    """
    token = _extract_bearer_token(authorization)

    ctx = resolve_common_context_from_token(
        token,
        db_override=(x_target_db or None),
        schema_override=(x_target_schema or None),
    )

    set_request_context(ctx)
    db = None

    try:
        schema = _validate_ident(ctx["schema"], "schema")
        db = get_request_session()

        row = db.execute(
            text(f'''
                SELECT 
                    model_blob,
                    model_name,
                    model_type,
                    model_version,
                    dependent_var,
                    features
                FROM "{schema}"."ai_trained_models"
                WHERE id = :model_id
            '''),
            {"model_id": model_id}
        ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"Model ID {model_id} not found")

        blob, name, mtype, version, dep_var, features_json = row

        if not blob:
            raise HTTPException(status_code=404, detail="Model blob is empty")

        # Parse features
        features = None
        if features_json:
            try:
                features = json.loads(features_json) if isinstance(features_json, str) else features_json
            except:
                features = features_json

        # Load the model from blob
        try:
            model_bundle = joblib.load(io.BytesIO(blob))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

        return {
            "success": True,
            "model_id": model_id,
            "model_name": name,
            "model_type": mtype,
            "model_version": version,
            "dependent_var": dep_var,
            "features": features or model_bundle.get("features", []),
            "blob_size": len(blob),
        }

    finally:
        if db is not None:
            db.close()
        clear_request_context()
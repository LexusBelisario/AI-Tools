from fastapi import APIRouter, Header, HTTPException, Form
from pathlib import Path
import geopandas as gpd
from sqlalchemy import text
import os
import json
from datetime import datetime
import re

from common_db_runtime import (
    connect_common_db,
    disconnect_common_db,
    get_common_db_meta,
    resolve_common_context_from_token,
)

router = APIRouter(prefix="/common", tags=["common"])


DATA_DIR = os.getenv("DATA_DIR", "/data").strip() or "/data"


def _safe_join_data_dir(p: str) -> str:
    """
    Docker-safe path handling with local development support.

    Development mode: accept absolute paths
    Docker mode: only accept paths inside DATA_DIR
    """
    if p is None:
        raise HTTPException(status_code=400, detail="Missing path")

    p = str(p).strip()
    if not p:
        raise HTTPException(status_code=400, detail="Empty path")

    if os.path.isabs(p) and os.path.exists(p):
        print(f"✅ Accepting absolute path (local dev mode): {p}")
        return p

    if os.path.isabs(p) and not os.path.exists(p):
        raise HTTPException(
            status_code=404,
            detail=f"Absolute path does not exist: {p}"
        )

    base = os.path.normpath(DATA_DIR)
    full = os.path.normpath(os.path.join(base, p))

    try:
        common = os.path.commonpath([base, full])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if common != base:
        raise HTTPException(status_code=400, detail="Invalid path (outside DATA_DIR)")

    return full


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


def _ensure_ai_trained_models_table(db, schema: str):
    schema = _validate_ident(schema, "schema")

    db.execute(text(f'''
        CREATE TABLE IF NOT EXISTS "{schema}"."ai_trained_models" (
            id BIGSERIAL PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
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
            ADD COLUMN IF NOT EXISTS metrics JSONB
    '''))

    db.commit()


@router.post("/connect")
def connect(
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

    return connect_common_db(ctx)


@router.get("/status")
def status(
    authorization: str = Header(default=""),
    x_target_schema: str = Header(default="", alias="X-Target-Schema"),
    x_target_db: str = Header(default="", alias="X-Target-DB"),
):
    if not authorization:
        return {"connected": False, "context": None, "meta": get_common_db_meta()}

    token = _extract_bearer_token(authorization)

    ctx = resolve_common_context_from_token(
        token,
        db_override=(x_target_db or None),
        schema_override=(x_target_schema or None),
    )

    return connect_common_db(ctx)


@router.post("/disconnect")
def disconnect():
    return disconnect_common_db()


@router.get("/meta")
def meta():
    return get_common_db_meta()


@router.post("/save-prediction-results")
async def save_prediction_results_to_common_db(
    shapefile_path: str = Form(...),
    model_type: str = Form(...),
    save_type: str = Form(...),
    authorization: str = Header(default=""),
    x_target_schema: str = Header(default="", alias="X-Target-Schema"),
):
    from common_db_runtime import (
        resolve_common_context_from_token,
        set_request_context,
        get_request_session,
        clear_request_context,
    )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization")

    token = authorization.split(" ", 1)[1].strip()
    db = None

    try:
        ctx = resolve_common_context_from_token(
            token,
            schema_override=(x_target_schema or None),
        )

        set_request_context(ctx)
        db = get_request_session()

        shapefile_path = _safe_join_data_dir(shapefile_path)
        shapefile_path_obj = Path(shapefile_path)
        if not shapefile_path_obj.exists():
            raise HTTPException(status_code=404, detail=f"Shapefile not found: {shapefile_path}")

        gdf = gpd.read_file(shapefile_path)
        if gdf.empty:
            raise HTTPException(status_code=400, detail="Shapefile is empty")

        schema = _validate_ident(ctx["schema"], "schema")
        if save_type == "training":
            table_name = f"training_predictions_{model_type}"
        else:
            table_name = "run_predictions"

        gdf.to_postgis(
            name=table_name,
            con=db.connection(),
            schema=schema,
            if_exists="replace",
            index=False,
        )

        db.commit()

        return {
            "success": True,
            "message": f'Saved to Common Database: "{schema}"."{table_name}"',
            "record_count": len(gdf),
            "table_name": f'"{schema}"."{table_name}"',
            "schema": schema,
            "model_type": model_type,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")

    finally:
        if db is not None:
            db.close()
        clear_request_context()


@router.post("/save-to-gis-db")
async def save_to_gis_database(
    shapefile_path: str = Form(...),
    schema: str = Form(...),
    model_type: str = Form(...),
    save_type: str = Form(...),
):
    from db import get_user_database_session

    db = None
    try:
        db = get_user_database_session()

        shapefile_path = _safe_join_data_dir(shapefile_path)
        shapefile_path_obj = Path(shapefile_path)
        if not shapefile_path_obj.exists():
            raise HTTPException(status_code=404, detail=f"Shapefile not found: {shapefile_path}")

        gdf = gpd.read_file(shapefile_path)
        if gdf.empty:
            raise HTTPException(status_code=400, detail="Shapefile is empty")

        schema = _validate_ident(schema, "schema")

        if save_type == "training":
            table_name = f"ai_training_{model_type}_results"
        else:
            table_name = "ai_run_results"

        gdf.to_postgis(
            name=table_name,
            con=db.connection(),
            schema=schema,
            if_exists="replace",
            index=False,
        )

        db.commit()

        return {
            "success": True,
            "message": f'Auto-saved to GIS Database: "{schema}"."{table_name}"',
            "record_count": len(gdf),
            "table_name": f'"{schema}"."{table_name}"',
            "schema": schema,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-save failed: {str(e)}")

    finally:
        if db is not None:
            db.close()


@router.post("/save-trained-model")
async def save_trained_model_to_common_db(
    model_path: str = Form(...),
    model_type: str = Form(...),
    dependent_var: str = Form(""),
    features_json: str = Form(""),
    metrics_json: str = Form(""),
    authorization: str = Header(default=""),
    x_target_schema: str = Header(default="", alias="X-Target-Schema"),
    x_target_db: str = Header(default="", alias="X-Target-DB"),
):
    from common_db_runtime import (
        resolve_common_context_from_token,
        set_request_context,
        get_request_session,
        clear_request_context,
    )

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization")

    token = authorization.split(" ", 1)[1].strip()

    ctx = resolve_common_context_from_token(
        token,
        db_override=(x_target_db or None),
        schema_override=(x_target_schema or None),
    )

    set_request_context(ctx)
    db = None

    try:
        schema = _validate_ident(ctx["schema"], "schema")

        model_path = _safe_join_data_dir(model_path)
        p = Path(model_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
        if p.suffix.lower() != ".pkl":
            raise HTTPException(status_code=400, detail="model_path must be a .pkl file")

        with open(str(p), "rb") as f:
            blob = f.read()

        try:
            features = json.loads(features_json) if features_json else None
        except Exception:
            features = None

        try:
            metrics = json.loads(metrics_json) if metrics_json else None
        except Exception:
            metrics = None

        db = get_request_session()
        _ensure_ai_trained_models_table(db, schema)

        meta = {
            "saved_from": "manual_save_button",
            "source_file": os.path.basename(str(p)),
            "saved_at": datetime.utcnow().isoformat(),
        }

        model_name = os.path.splitext(os.path.basename(str(p)))[0]

        row = db.execute(
            text(f'''
                INSERT INTO "{schema}"."ai_trained_models"
                    (model_name, model_type, dependent_var, features, metrics, model_blob, meta)
                VALUES
                    (:model_name, :model_type, :dependent_var,
                     CAST(:features AS JSONB), CAST(:metrics AS JSONB), :model_blob, CAST(:meta AS JSONB))
                RETURNING id
            '''),
            {
                "model_name": model_name,
                "model_type": model_type,
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
            "message": f"Saved trained model to Common DB table {schema}.ai_trained_models",
            "schema": schema,
            "id": new_id,
            "model_name": model_name,
        }

    finally:
        if db is not None:
            db.close()
        clear_request_context()


@router.post("/auto-save-training-results")
async def auto_save_training_results(
    model_path: str = Form(...),
    shapefile_path: str = Form(...),
    model_type: str = Form(...),
    dependent_var: str = Form(""),
    features_json: str = Form("[]"),
    metrics_json: str = Form("{}"),
    authorization: str = Header(default=""),
    x_target_schema: str = Header(default="", alias="X-Target-Schema"),
    x_target_db: str = Header(default="", alias="X-Target-DB"),
):
    from common_db_runtime import (
        resolve_common_context_from_token,
        set_request_context,
        get_request_session,
        clear_request_context,
    )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization")

    token = authorization.split(" ", 1)[1].strip()
    db = None

    try:
        ctx = resolve_common_context_from_token(
            token,
            db_override=(x_target_db or None),
            schema_override=(x_target_schema or None),
        )

        set_request_context(ctx)
        db = get_request_session()
        schema = _validate_ident(ctx["schema"], "schema")

        print("🔍 Validating paths...")
        print(f"   Model path (raw): {model_path}")
        print(f"   Shapefile path (raw): {shapefile_path}")

        model_path = _safe_join_data_dir(model_path)
        shapefile_path = _safe_join_data_dir(shapefile_path)

        print(f"   ✅ Model path (validated): {model_path}")
        print(f"   ✅ Shapefile path (validated): {shapefile_path}")

        model_saved_id = None
        model_name = None

        if model_path and os.path.exists(model_path):
            _ensure_ai_trained_models_table(db, schema)

            with open(model_path, "rb") as f:
                model_blob = f.read()

            try:
                features = json.loads(features_json) if features_json else None
            except Exception:
                features = None

            try:
                metrics = json.loads(metrics_json) if metrics_json else None
            except Exception:
                metrics = None

            meta = {
                "saved_from": "auto_save_after_training",
                "source_file": os.path.basename(model_path),
                "saved_at": datetime.utcnow().isoformat(),
            }

            model_name = os.path.splitext(os.path.basename(model_path))[0]

            row = db.execute(
                text(f'''
                    INSERT INTO "{schema}"."ai_trained_models"
                        (model_name, model_type, dependent_var, features, metrics, model_blob, meta)
                    VALUES
                        (:model_name, :model_type, :dependent_var,
                         CAST(:features AS JSONB), CAST(:metrics AS JSONB), :model_blob, CAST(:meta AS JSONB))
                    RETURNING id
                '''),
                {
                    "model_name": model_name,
                    "model_type": model_type,
                    "dependent_var": dependent_var or None,
                    "features": json.dumps(features) if features is not None else None,
                    "metrics": json.dumps(metrics) if metrics is not None else None,
                    "model_blob": model_blob,
                    "meta": json.dumps(meta),
                },
            ).fetchone()

            model_saved_id = int(row[0]) if row else None
            db.commit()
            print(f"✅ Saved model to database (ID: {model_saved_id}) as {model_name}")

        predictions_saved = False
        prediction_count = 0
        predictions_table = None

        if shapefile_path and os.path.exists(shapefile_path):
            print("📊 Loading shapefile for predictions...")
            gdf = gpd.read_file(shapefile_path)
            if not gdf.empty:
                import hashlib
                predictions_table = model_name or f"training_predictions_{model_type}"
                gdf.to_postgis(
                    name=predictions_table,
                    con=db.connection(),
                    schema=schema,
                    if_exists="replace",
                    index=False,
                )
                # Manually create spatial index with a short hash-based name
                # to avoid PostgreSQL's 63-char identifier limit on auto-generated index names
                geom_col = gdf.geometry.name
                idx_hash = hashlib.md5(predictions_table.encode()).hexdigest()[:8]
                idx_name = f"idx_{idx_hash}_geom"
                db.execute(text(f'''
                    CREATE INDEX IF NOT EXISTS "{idx_name}"
                    ON "{schema}"."{predictions_table}"
                    USING gist ("{geom_col}")
                '''))
                db.commit()
                predictions_saved = True
                prediction_count = len(gdf)
                print(f"✅ Saved {prediction_count} predictions to {predictions_table}")

        return {
            "success": True,
            "message": f"Auto-saved {model_type.upper()} to Common Database",
            "schema": schema,
            "model_id": model_saved_id,
            "model_name": model_name,
            "predictions_saved": predictions_saved,
            "prediction_count": prediction_count,
            "model_table": "ai_trained_models",
            "predictions_table": f"{schema}.{predictions_table}" if predictions_saved else None,
        }

    except Exception as e:
        import traceback
        print("❌ AUTO-SAVE ERROR:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Auto-save failed: {str(e)}")

    finally:
        if db is not None:
            db.close()
        clear_request_context()
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, FileResponse
import geopandas as gpd
import tempfile, os, json, zipfile
import matplotlib
matplotlib.use("Agg")
from db import get_user_database_session
from .lr_train import get_provincial_code_from_schema
import urllib.parse, shutil

router = APIRouter()


@router.get("/download")
async def download_file(file: str = Query(...)):
    try:
        if not os.path.exists(file):
            return JSONResponse(status_code=404, content={"error": "File not found."})
        filename = os.path.basename(file)
        return FileResponse(path=file, filename=filename, media_type="application/octet-stream")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    
@router.get("/predicted-geojson")
def get_predicted_geojson(table: str = "Predicted_Output", schema: str = "public"):
    try:
        provincial_code = get_provincial_code_from_schema(schema)
        db_session = get_user_database_session(provincial_code)
        engine = db_session.get_bind()
        sql = f'SELECT * FROM "{schema}"."{table}"'
        gdf = gpd.read_postgis(sql, engine, geom_col="geometry")
        engine.dispose()
        return json.loads(gdf.to_json())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/save-to-db")
async def save_to_db(payload: dict):
    shapefile_url = payload.get("shapefile_url")
    table_name = payload.get("table_name", "Predicted_Output")
    if not shapefile_url or not shapefile_url.endswith(".zip"):
        return JSONResponse(status_code=400, content={"error": "Invalid shapefile URL."})
    try:
        file_path = shapefile_url.split("?file=")[-1]
        if not os.path.exists(file_path):
            return JSONResponse(status_code=404, content={"error": "Shapefile not found."})

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(file_path, "r") as z:
                z.extractall(tmpdir)
            shp_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")]
            if not shp_files:
                return JSONResponse(status_code=400, content={"error": "No .shp file found in zip."})
            shp_path = shp_files[0]
            gdf = gpd.read_file(shp_path)

            schema = payload.get("schema", "public")
            provincial_code = get_provincial_code_from_schema(schema)
            db_session = get_user_database_session(provincial_code)
            engine = db_session.get_bind()

            gdf.to_postgis(
                name=table_name,
                con=engine,
                schema=schema,
                if_exists="replace",
                index=False
            )
            engine.dispose()
        return {"message": f"Saved successfully to {schema}.{table_name}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/preview-geojson")
def preview_geojson(file_path: str):
    try:
        if file_path.startswith("/api/linear-regression/download"):
            parsed = urllib.parse.urlparse(file_path)
            query_params = urllib.parse.parse_qs(parsed.query)
            file_path = query_params.get("file", [None])[0]
            if not file_path:
                return JSONResponse({"error": "Invalid file parameter in URL."}, status_code=400)
            file_path = urllib.parse.unquote(file_path)

        file_path = file_path.strip('"').strip("'")
        if not os.path.exists(file_path):
            return JSONResponse({"error": f"File not found: {file_path}"}, status_code=404)

        tmpdir = tempfile.mkdtemp()
        extract_dir = os.path.join(tmpdir, "unzipped")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        shp_file = next(
            (os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith(".shp")),
            None,
        )
        if not shp_file:
            shutil.rmtree(tmpdir, ignore_errors=True)
            return JSONResponse({"error": "No .shp found inside ZIP."}, status_code=400)

        gdf = gpd.read_file(shp_file)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        geojson_data = json.loads(gdf.to_json())
        shutil.rmtree(tmpdir, ignore_errors=True)
        return geojson_data
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
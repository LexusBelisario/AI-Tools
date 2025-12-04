from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, JSONResponse
import geopandas as gpd
import tempfile, os, json, zipfile
import matplotlib

matplotlib.use("Agg")

from db import get_user_database_session
from AITools.ai_utils import get_provincial_code_from_schema
import urllib.parse, shutil

router = APIRouter()

@router.get("/download")
async def ai_download(file: str = Query(..., description="Absolute path to file on server")):
    """
    Generic download endpoint.

    Frontend calls:
      /api/ai-tools/download?file=C:\...\RandomForest_CAMA_....csv
    (the /api prefix is from Vite proxy only)
    """
    try:
        if not os.path.exists(file):
            return JSONResponse(status_code=404, content={"error": "File not found"})
        filename = os.path.basename(file)
        return FileResponse(path=file, filename=filename)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/predicted-geojson")
def get_predicted_geojson(
    table: str = "Predicted_Output",
    schema: str = "public",
):
    """
    Read a predicted table from the DB (with geometry column 'geometry')
    and return as GeoJSON. Generic — works for any table/schema.
    """
    try:
        provincial_code = get_provincial_code_from_schema(schema)
        db_session = get_user_database_session(provincial_code)
        engine = db_session.get_bind()

        sql = f'SELECT * FROM "{schema}"."{table}"'
        gdf = gpd.read_postgis(sql, engine, geom_col="geometry")

        # Ensure WGS84 for web map
        if gdf.crs:
            if gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
        else:
            gdf.set_crs(epsg=4326, inplace=True)

        engine.dispose()
        db_session.close()

        return json.loads(gdf.to_json())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/save-to-db")
async def save_to_db(payload: dict):
    """
    Given a shapefile ZIP URL (same style as /download?file=...),
    import the shapefile into PostGIS as schema.table_name.
    """
    shapefile_url = payload.get("shapefile_url")
    table_name = payload.get("table_name", "Predicted_Output")
    schema = payload.get("schema", "public")

    if not shapefile_url or not shapefile_url.endswith(".zip"):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid shapefile URL."},
        )

    try:
        # Decode URL and extract the real file path after ?file=
        decoded = urllib.parse.unquote(shapefile_url)
        if "file=" in decoded:
            file_path = decoded.split("file=")[-1]
        else:
            file_path = decoded

        file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            return JSONResponse(
                status_code=404,
                content={"error": f"Shapefile not found: {file_path}"},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(file_path, "r") as z:
                z.extractall(tmpdir)

            # Find the first .shp inside the extracted folder
            shp_files = [
                os.path.join(root, f)
                for root, _, files in os.walk(tmpdir)
                for f in files
                if f.lower().endswith(".shp")
            ]
            if not shp_files:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No .shp file found in zip."},
                )

            shp_path = shp_files[0]

            # Read and reproject to EPSG:4326 if needed
            gdf = gpd.read_file(shp_path)
            if gdf.crs:
                if gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)
            else:
                gdf.set_crs(epsg=4326, inplace=True)

            # Save to PostGIS
            provincial_code = get_provincial_code_from_schema(schema)
            db_session = get_user_database_session(provincial_code)
            engine = db_session.get_bind()

            gdf.to_postgis(
                name=table_name,
                con=engine,
                schema=schema,
                if_exists="replace",
                index=False,
            )

            engine.dispose()
            db_session.close()

        return {
            "message": f"✅ Shapefile saved to {schema}.{table_name}",
            "schema": schema,
            "table": table_name,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/preview-geojson")
async def ai_preview_geojson(
    file_path: str = Query(..., description="Absolute path to shapefile ZIP")
):
    """
    Frontend calls:
      /api/ai-tools/preview-geojson?file_path=C:\...\RandomForest_Predicted.zip
    """
    try:
        if not os.path.exists(file_path):
            return JSONResponse(status_code=404, content={"error": "File not found"})

        src = file_path

        # If ZIP, extract to temp and read first .shp
        if file_path.lower().endswith(".zip"):
            tmpdir = tempfile.mkdtemp()
            with zipfile.ZipFile(file_path, "r") as z:
                z.extractall(tmpdir)
            shp_candidates = [
                os.path.join(tmpdir, f)
                for f in os.listdir(tmpdir)
                if f.lower().endswith(".shp")
            ]
            if not shp_candidates:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No .shp file found inside ZIP."},
                )
            src = shp_candidates[0]

        gdf = gpd.read_file(src)
        return json.loads(gdf.to_json())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
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
async def download_file(file: str = Query(...)):
    """
    Serve generated model/report/shapefile for download.
    Same behavior as LR download.
    """
    try:
        if not os.path.exists(file):
            return JSONResponse(
                status_code=404,
                content={"error": "File not found."},
            )
        filename = os.path.basename(file)
        return FileResponse(
            path=file,
            filename=filename,
            media_type="application/octet-stream",
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/predicted-geojson")
def get_predicted_geojson(
    table: str = "Predicted_Output",
    schema: str = "public",
):
    """
    Read a predicted table from the DB (with geometry column 'geometry')
    and return as GeoJSON. Generic — works for any table/scheme.
    """
    try:
        provincial_code = get_provincial_code_from_schema(schema)
        db_session = get_user_database_session(provincial_code)
        engine = db_session.get_bind()

        sql = f'SELECT * FROM "{schema}"."{table}"'
        gdf = gpd.read_postgis(sql, engine, geom_col="geometry")

        engine.dispose()
        db_session.close()

        return json.loads(gdf.to_json())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/save-to-db")
async def save_to_db(payload: dict):
    """
    Given a shapefile ZIP URL (the same style used by /download?file=...),
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

        if not os.path.exists(file_path):
            return JSONResponse(
                status_code=404,
                content={"error": "Shapefile not found."},
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
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)

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
async def preview_geojson(file: str = Query(...)):
    """
    Given a ?file=... (ZIP path like from /download), open the ZIP,
    find the first .shp, and return it as GeoJSON (EPSG:4326).
    Same as LR preview-geojson.
    """
    try:
        decoded = urllib.parse.unquote(file)
        if "file=" in decoded:
            zip_path = decoded.split("file=")[-1]
        else:
            zip_path = decoded

        if not os.path.exists(zip_path):
            return JSONResponse(
                {"error": f"File not found: {zip_path}"},
                status_code=404,
            )

        tmpdir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmpdir)

            shp_file = next(
                (
                    os.path.join(root, f)
                    for root, _, files in os.walk(tmpdir)
                    for f in files
                    if f.lower().endswith(".shp")
                ),
                None,
            )
            if not shp_file:
                shutil.rmtree(tmpdir, ignore_errors=True)
                return JSONResponse(
                    {"error": "No .shp found inside ZIP."},
                    status_code=400,
                )

            gdf = gpd.read_file(shp_file)
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)

            geojson_data = json.loads(gdf.to_json())
            shutil.rmtree(tmpdir, ignore_errors=True)
            return geojson_data
        except Exception as e:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise e
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

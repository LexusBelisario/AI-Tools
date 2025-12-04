from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import geopandas as gpd

router = APIRouter()

@router.get("/preview-geojson")
async def preview_geojson(file_path: str = Query(...)):
    try:
        gdf = gpd.read_file(file_path)
        return JSONResponse(gdf.to_json())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

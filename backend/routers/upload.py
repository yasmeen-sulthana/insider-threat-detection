import os
import shutil

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

router = APIRouter()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_files(
    device: UploadFile = File(None),
    logon: UploadFile = File(None),
    file: UploadFile = File(None),
    email: UploadFile = File(None),
    http: UploadFile = File(None),
):

    uploaded = {}

    try:

        files = {
            "device": device,
            "logon": logon,
            "file": file,
            "email": email,
            "http": http,
        }

        for key, uploaded_file in files.items():

            if uploaded_file is not None:

                save_path = os.path.join(
                    UPLOAD_DIR,
                    f"{key}.csv"
                )

                with open(save_path, "wb") as buffer:
                    shutil.copyfileobj(
                        uploaded_file.file,
                        buffer
                    )

                uploaded[key] = True

            else:

                uploaded[key] = False

        return JSONResponse({
            "success": True,
            "uploaded": uploaded
        })

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@router.get("/upload/status")
async def upload_status():

    status = {}

    for name in [
        "device",
        "logon",
        "file",
        "email",
        "http"
    ]:

        path = os.path.join(
            UPLOAD_DIR,
            f"{name}.csv"
        )

        status[name] = os.path.exists(path)

    return status
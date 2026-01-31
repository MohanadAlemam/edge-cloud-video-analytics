#-----------------------------------------------------------------------------------------------------------------------
# A function to feed the could frames
#-----------------------------------------------------------------------------------------------------------------------
import requests

def feed_cloud_jpeg(
        cloud_server_url:str,
        jpeg_bytes:bytes,
        requests_timeout_sec: int =10
):
    """
    Send the in-memory jpeg bytes to cloud server for inference.

    it only does the HTTP request and return parsed JSON.
    """
    # prepare multipart or form data payload, field "image" with (filename, bytes, mime-type/ media type eg JPEG image)
    files_payload = {
        "image": ("frame.jpg", jpeg_bytes, "image/jpeg")
    }
    # will post the in memory JPEG as a multi-part file/payload to server/inference
    response = requests.post(cloud_server_url.rstrip("/") + "/inference",
                             files=files_payload, timeout=requests_timeout_sec)
    # normalize URL by removing slashes if present and append the /infer endpoint then POST files

    response.raise_for_status()
    # Error handling for HTTP requests. to avoid failed request or errors (e.g. 500, 404, 401) being treated as inference

    # parse response body as JSON and return the resulting Python object e.g. dictionary
    return response.json()
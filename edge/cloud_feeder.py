import requests
# HTTP client-server communication, sends data to the server/cloud

def feed_cloud_jpeg(
        cloud_server_url:str,
        jpeg_bytes:bytes,
        requests_timeout_sec: int =10
):
    """
    Send the in-memory jpeg bytes to cloud server for inference.

    it only conducts the HTTP requests and return parsed JSON.
    """
    # prepare multipart or form data payload
    file_payload = {
        "image": ("frame.jpg", jpeg_bytes, "image/jpeg")
    }
    # post multi-part payload to server/inference
    response = requests.post(cloud_server_url.rstrip("/") + "/infer",
                             files=file_payload, timeout=requests_timeout_sec)
    # normalize URL
    response.raise_for_status()
    # Error handling for HTTP requests. to avoid failed request or errors (e.g 500, 404, 401) being treated as inference

    # parse response body as JSON
    return response.json()
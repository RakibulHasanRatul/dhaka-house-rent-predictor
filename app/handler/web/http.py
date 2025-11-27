import datetime
import json
from http.server import BaseHTTPRequestHandler

from config import LOCATION_JSON_DIR, TYPES_JSON_DIR, UI_DIR

from ...helper import construct_features_list
from ...predict_rent import predict_rent

year = datetime.datetime.now().year


class HttpHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            try:
                with open(f"{UI_DIR}/index.html", "rb") as index_html:
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()

                    self.wfile.write(index_html.read())
            except FileNotFoundError:
                self.send_response(404)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"404 Not Found: index.html")

        if self.path == "/locations":
            # not adding error handling logics, I assume user will train first then serve the ui!
            with open(LOCATION_JSON_DIR, "r") as file:
                res: dict[str, list[str]] = json.load(file)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        if self.path == "/types":
            # again, not adding error handling logics, I assume user will train first then serve the ui!
            with open(TYPES_JSON_DIR, "r") as file:
                to_response: dict[str, list[str]] = json.load(file)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(to_response).encode("utf-8"))

    def do_POST(self) -> None:
        if self.path == "/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                request_body = self.rfile.read(content_length).decode("utf-8")

            else:
                self.send_response(400)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"400 Bad Request: No data received")
                return

            if (
                "Content-Type" not in self.headers
                and self.headers["Content-Type"] != "application/json"
            ):
                self.send_response(400)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(
                    b"400 Bad Request: Content-Type must be application/json"
                )
                return
            try:
                data = json.loads(request_body)
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"400 Bad Request: Invalid JSON format")
                return

            # Check if the required keys are present in the JSON data
            required_keys = [
                "location",
                "type",
                "floorArea",
                "numberOfWashrooms",
                "numberOfBedrooms",
            ]
            for key in required_keys:
                if key not in data:
                    self.send_response(400)
                    self.send_header("Content-type", "text/plain")
                    self.end_headers()
                    self.wfile.write(
                        f"400 Bad Request: Missing key '{key}' in JSON data".encode(
                            "utf-8"
                        )
                    )
                    return

            with open(TYPES_JSON_DIR, "r") as file:
                rent_types: list[str] = json.load(file)["types"]

            # extract values
            location = data["location"]
            type_num = rent_types.index(data["type"])
            floor_area = data["floorArea"]
            number_of_washrooms = data["numberOfWashrooms"]
            number_of_bedrooms = data["numberOfBedrooms"]

            print(
                f"INFO: Predicting using {location=}, {type_num=}, {floor_area=}, {number_of_washrooms=}, {number_of_bedrooms=}, {year=}"
            )

            prediction = predict_rent(
                location,
                *construct_features_list(
                    number_of_bedrooms,
                    number_of_washrooms,
                    floor_area,
                    type_num,
                    year,
                ),
            )

            print(f"INFO: Prediction: {prediction:.5f}")

            # Send the response
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            to_response = {"prediction": round(prediction)}
            self.wfile.write(json.dumps(to_response).encode("utf-8"))

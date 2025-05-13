from app import app  # noqa: F401
import routes  # noqa: F401
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, jsonify, render_template, request

from governance.controller import GovernanceController

app = Flask(__name__)
controller = GovernanceController()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_clause():
    payload = request.get_json(silent=True) or {}
    clause = payload.get("clause")

    if not clause and request.form:
        clause = request.form.get("clause")

    if not clause or not clause.strip():
        return jsonify({"error": "Please provide a contract clause."}), 400

    result = controller.run_pipeline(clause.strip())
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

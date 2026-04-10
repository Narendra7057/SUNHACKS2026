from flask import Flask, jsonify, render_template, request

from governance.controller import GovernanceController
from agents.domain_agent_factory import DomainAgentFactory

app = Flask(__name__)
controller = GovernanceController()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_clause():
    try:
        payload = request.get_json(silent=True) or {}
        domain = payload.get("domain", "legal").lower()
        input_data = payload.get("input") or payload.get("clause")

        if not input_data and request.form:
            input_data = request.form.get("input") or request.form.get("clause")
            domain = request.form.get("domain", "legal").lower()

        if not input_data or not input_data.strip():
            return jsonify({"error": "Please provide input data for analysis."}), 400

        if not DomainAgentFactory.is_valid_domain(domain):
            return jsonify({"error": f"Invalid domain. Valid options: legal, ecommerce, fintech, healthcare"}), 400

        result = controller.run_pipeline(input_data.strip(), domain)
        return jsonify(result)
    except Exception as e:
        error_msg = str(e)
        if "Ollama" in error_msg or "endpoint" in error_msg.lower():
            return jsonify({
                "error": "Ollama is not running. Please start Ollama and ensure the mistral model is available.",
                "details": error_msg
            }), 503
        return jsonify({"error": f"Analysis failed: {error_msg}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

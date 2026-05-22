from flask import Flask, request, jsonify

from services.llm_service import generate_answer, clear_memory, check_model
from services.ingest_service import ingest_docs

app = Flask(__name__)


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "No question provided"}), 400

        answer, sources = generate_answer(question)

        return jsonify({
            "question": question,
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ingest", methods=["POST"])
def ingest():
    ingest_docs()
    return jsonify({"status": "ingested"})


@app.route("/clear", methods=["POST"])
def clear():
    clear_memory()
    return jsonify({"status": "cleared"})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/model-health")
def model_health():
    return jsonify({"ollama": "ok" if check_model() else "down"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
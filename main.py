# main.py

from flask import Flask, request, jsonify, send_file
from simulation import CheckpointedStabilizerSimulation
import os

app = Flask(__name__)
sim = CheckpointedStabilizerSimulation()

@app.route("/")
def index():
    return jsonify({"message": "TUNLOmega2 Checkpoint API running."})

@app.route("/start", methods=["POST"])
def start_sim():
    params = request.get_json() or {}
    sim.reset_simulation(params_override=params)
    sim.run()
    return jsonify({"status": "started", "checkpoint_dir": sim.checkpoint_dir})

@app.route("/resume", methods=["POST"])
def resume_sim():
    sim.resume_checkpoint()
    sim.run()
    return jsonify({"status": "resumed", "checkpoint_dir": sim.checkpoint_dir})

@app.route("/status", methods=["GET"])
def status():
    meta_file, last_chunk = sim.get_latest_checkpoint_files()
    meta = None
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            meta = f.read()
    return jsonify({
        "checkpoint_meta": meta,
        "last_chunk": last_chunk,
        "running": True
    })

@app.route("/download_chunk", methods=["GET"])
def download_chunk():
    phase = int(request.args.get("phase", 0))
    fn = sim.checkpoint_filename(phase)
    if not os.path.exists(fn):
        return jsonify({"error": f"Chunk {phase} not found"}), 404
    return send_file(fn, as_attachment=True)

@app.route("/list_chunks", methods=["GET"])
def list_chunks():
    files = [f for f in os.listdir(sim.checkpoint_dir) if f.endswith(".jsonl")]
    return jsonify({"chunks": files})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000)

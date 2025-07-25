from flask import Flask, request, jsonify
# import flask_cors
from module import agentic_rag_demo
import uuid

app = Flask(__name__)

thread_id = None

@app.route("/run", methods=['POST'])
def run_agent():
    global thread_id
    usr_in = request.form["prompt"]
    if not thread_id:
        thread_id = str(uuid.uuid4())
    if usr_in:
        try:
            resp = agentic_rag_demo.run_chat(usr_in, thread_id)
            return f'''
            <form method="POST">
            <label>Prompt:</label>
            <input name= "prompt"><br><br>
            <input type="submit">
            </form>
            <p>Response: {resp}</p>'''
        except Exception as e:
            return f'''
            <form method="POST">
            <label>Prompt:</label>
            <input name= "prompt"><br><br>
            <input type="submit">
            </form>
            <p>Response: Error occurred - {str(e)}</p>'''

@app.route("/exit", methods=['POST'])
def exit_agent():
    global thread_id
    thread_id = None
    return '''
    <p>Session ended. You can start a new session.</p>
    '''

    
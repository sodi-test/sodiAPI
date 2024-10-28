from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import time
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import datetime
from threading import Lock

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

openai_client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    default_headers={"OpenAI-Beta": "assistants=v2"}
)

ASSISTANT_ID = os.getenv('ASSISTANT_ID')

# Conexión a MongoDB Atlas
app.logger.info(f"Intentando conectar a MongoDB Atlas con URI: {os.getenv('MONGO_URI')}")
try:
    mongo_client = MongoClient(
        os.getenv('MONGO_URI'),
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=5000
    )
    # Intenta una operación simple para verificar la conexión
    mongo_client.admin.command('ping')
    db = mongo_client['ChatbotBD']
    conversations = db['conversations']
    errors = db['errors']
    app.logger.info("Conexión exitosa a MongoDB Atlas")
except Exception as e:
    app.logger.error(f"Error al conectar a MongoDB Atlas: {str(e)}")
    mongo_client = None

thread_locks = {}

@app.route('/')
def index():
    return jsonify({
        "message": "API is running",
        "endpoints": {
            "test": "/test",
            "start_conversation": "/start_conversation",
            "send_message": "/send_message",
            "get_conversation": "/get_conversation"
        }
    })

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    try:
        thread = openai_client.beta.threads.create()
        thread_id = thread.id
        return jsonify({'thread_id': thread_id})
    except Exception as e:
        app.logger.error(f"Error al iniciar conversación: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    thread_id = data.get('thread_id')
    user_message = data.get('message')

    if thread_id not in thread_locks:
        thread_locks[thread_id] = Lock()

    with thread_locks[thread_id]:
        try:
            # Verificar si hay una ejecución activa
            runs = openai_client.beta.threads.runs.list(thread_id=thread_id)
            active_run = next((run for run in runs.data if run.status == 'in_progress'), None)
            
            if active_run:
                return jsonify({'error': 'Hay una solicitud en proceso. Por favor, espera.'}), 409

            app.logger.info(f"Received request: thread_id={thread_id}, message={user_message}")

            if not thread_id or not user_message:
                app.logger.error("thread_id or message is missing")
                return jsonify({'error': 'thread_id and message are required'}), 400

            try:
                # Añadir el mensaje del usuario al hilo
                openai_client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=user_message
                )
                
                # Crear un nuevo run para obtener la respuesta del asistente
                run = openai_client.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=ASSISTANT_ID
                )
                
                # Esperar a que el run se complete con un timeout manual
                start_time = time.time()
                timeout_duration = 30  # segundos
                while True:
                    if time.time() - start_time > timeout_duration:
                        # Cancelar el run si el tiempo de espera se excede
                        openai_client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                        raise TimeoutError("Timeout occurred while waiting for the API response")
                    
                    run_status = openai_client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
                    if run_status.status == 'completed':
                        break
                    elif run_status.status in ['failed', 'cancelled', 'expired']:
                        app.logger.error(f"Run failed with status: {run_status.status}")
                        raise Exception(f"Run failed with status: {run_status.status}")
                    time.sleep(1)

                # Obtener la respuesta más reciente del asistente
                messages = openai_client.beta.threads.messages.list(thread_id=thread_id)
                assistant_message = next((msg for msg in messages.data if msg.role == "assistant"), None)
                
                if assistant_message:
                    assistant_response = assistant_message.content[0].text.value
                    # Guardar la conversación en la base de datos
                    if not save_conversation_to_db(thread_id, user_message, assistant_response):
                        app.logger.error("Error al guardar la conversación en la base de datos")
                    return jsonify({'response': assistant_response, 'thread_id': thread_id})
                else:
                    app.logger.error("No se pudo obtener una respuesta del asistente")
                    return jsonify({'response': 'No se pudo obtener una respuesta.', 'thread_id': thread_id})
            except TimeoutError as e:
                error_message = str(e)
                app.logger.error(error_message)
                save_error_to_db(thread_id, user_message, error_message)
                return jsonify({'error': 'La solicitud ha tardado demasiado. Por favor, intenta reformular tu pregunta.'}), 504
            except Exception as e:
                error_message = str(e)
                app.logger.error(f"Error al enviar mensaje: {error_message}", exc_info=True)
                save_error_to_db(thread_id, user_message, error_message)
                return jsonify({'error': error_message}), 500
        except Exception as e:
            app.logger.error(f"Error al enviar mensaje: {str(e)}")
            return jsonify({'error': str(e)}), 500

@app.route('/get_conversation', methods=['GET'])
def get_conversation():
    thread_id = request.args.get('thread_id')
    try:
        messages = openai_client.beta.threads.messages.list(thread_id=thread_id)
        return jsonify({'messages': [{'role': m.role, 'content': m.content[0].text.value} for m in messages.data]})
    except Exception as e:
        app.logger.error(f"Error al obtener conversación: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/end_conversation', methods=['POST'])
def end_conversation():
    # Aquí podrías implementar lógica para "cerrar" el hilo si es necesario
    return jsonify({'message': 'Conversación finalizada'})

@app.errorhandler(Exception)
def handle_exception(e):
    # Registra el error
    app.logger.error(f"Error no manejado: {str(e)}")
    # Devuelve una respuesta JSON con detalles del error
    return jsonify(error=str(e)), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test successful"}), 200


def save_conversation_to_db(thread_id, user_message, assistant_response):
    if mongo_client is None:
        app.logger.error("No hay conexión a MongoDB Atlas. No se puede guardar la conversación.")
        return False
    try:
        now = datetime.datetime.now()
        conversation = conversations.find_one({'thread_id': thread_id})
        if conversation:
            conversations.update_one(
                {'thread_id': thread_id},
                {
                    '$push': {
                        'messages': {
                            '$each': [
                                {'role': 'user', 'content': user_message, 'timestamp': now},
                                {'role': 'assistant', 'content': assistant_response, 'timestamp': now}
                            ]
                        }
                    },
                    '$set': {'updated_at': now}
                }
            )
        else:
            conversations.insert_one({
                'thread_id': thread_id,
                'messages': [
                    {'role': 'user', 'content': user_message, 'timestamp': now},
                    {'role': 'assistant', 'content': assistant_response, 'timestamp': now}
                ],
                'created_at': now,
                'updated_at': now
            })
        return True
    except Exception as e:
        app.logger.error(f"Error al insertar conversación en MongoDB Atlas: {e}")
        return False

def save_error_to_db(thread_id, user_message, error_message):
    if mongo_client is None:
        app.logger.error("No hay conexión a MongoDB Atlas. No se puede guardar el error.")
        return False
    try:
        now = datetime.datetime.now()
        errors.insert_one({
            'thread_id': thread_id,
            'user_message': user_message,
            'error_message': error_message,
            'timestamp': now
        })
        return True
    except Exception as e:
        app.logger.error(f"Error al insertar error en MongoDB Atlas: {e}")
        return False

app = app

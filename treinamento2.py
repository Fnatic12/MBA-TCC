import glob
import os
import sys
import random
import time
import numpy as np
import math
import tensorflow as tf
from collections import deque
from tqdm import tqdm
from keras.applications import Xception, MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

# Escolha do modelo
MODEL_TYPE = "Xception"  # Altere para "MobileNetV2" para testar outro modelo

# 📌 Configuração do TensorFlow para GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 📌 Importando o CARLA
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    import carla
    print("✅ Módulo CARLA importado com sucesso!")
except IndexError:
    print("🚫 ERRO: Falha ao importar o módulo CARLA. Verifique o caminho do .egg.")
    sys.exit(1)

# 🔹 Configurações do ambiente
IM_WIDTH = 320  # 🔹 Reduzindo a resolução para otimizar desempenho
IM_HEIGHT = 240
REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 32
DISCOUNT = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 1000
FPS = 20  # 🔹 Ajustado para maior estabilidade
MODEL_NAME = f"{MODEL_TYPE}_DQN"

# 🔹 Criando diretório para logs do TensorBoard
LOG_DIR = f"logs_treinamento_{MODEL_TYPE}"
os.makedirs(LOG_DIR, exist_ok=True)
writer = tf.summary.create_file_writer(LOG_DIR)

# ✅ Ambiente CARLA
class CarEnv:
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(35.0)
        print("🔄 Conectando ao CARLA...")
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.actor_list = []
        self.front_camera = None
        self.collision_hist = []
        self.total_distance = 0  

        # 🔹 Configuração do ambiente para melhor desempenho
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = True  # Desativando renderização
        self.settings.fixed_delta_seconds = 1.0 / FPS
        self.world.apply_settings(self.settings)
        print("✅ Ambiente configurado!")

    def reset(self):
        """Reseta o ambiente e inicia um novo episódio"""
        print("🔄 Resetando ambiente...")
        self.clear_actors()
        self.collision_hist = []
        self.total_distance = 0  

        for _ in range(5):  # 🔹 Evita erro ao spawnar
            try:
                spawn_point = random.choice(self.world.get_map().get_spawn_points())
                self.vehicle = self.world.spawn_actor(self.model_3, spawn_point)
                if self.vehicle:
                    print("✅ Veículo spawnado!")
                    break
            except:
                time.sleep(1)

        if not self.vehicle:
            raise RuntimeError("🚨 Falha ao spawnar veículo!")

        self.actor_list.append(self.vehicle)

        # 🔹 Sensor de câmera
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.rgb_cam.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.rgb_cam.set_attribute("fov", "110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        start_time = time.time()
        while self.front_camera is None:
            if time.time() - start_time > 5:
                print("🚨 ERRO: Câmera não recebeu imagens!")
                break
            time.sleep(0.01)

        print("✅ Reset concluído!")
        return self.front_camera

    def process_img(self, image):
        """Processa a imagem da câmera"""
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        self.front_camera = i2[:, :, :3]

    def clear_actors(self):
        """Remove sensores e veículos anteriores"""
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []

# ✅ Modelo DQN
class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self):
        if MODEL_TYPE == "Xception":
            base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))
        else:
            base_model = MobileNetV2(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))
        
        x = GlobalAveragePooling2D()(base_model.output)
        predictions = Dense(5, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
        return model

# ✅ Execução do treinamento
if __name__ == '__main__':
    env = CarEnv()
    agent = DQNAgent()
    
    print(f"🚀 Iniciando treinamento com {MODEL_TYPE}...")
    
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episódios'):
        print(f"\n🎬 Episódio {episode}/{EPISODES}")
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        num_colisoes = 0
        velocidade_acumulada = 0
        recompensas = []

        while not done:
            action = np.random.randint(0, 5)
            new_state, reward, done, kmh = env.step(action)

            recompensas.append(reward)
            velocidade_acumulada += kmh
            if done:
                num_colisoes += 1

            total_reward += reward
            steps += 1
            state = new_state

        # 🔹 Cálculo das métricas
        velocidade_media = velocidade_acumulada / max(1, steps)  
        tempo_sobrevivencia = steps / FPS  
        distancia_percorrida = velocidade_media * tempo_sobrevivencia  
        recompensa_media = np.mean(recompensas) if recompensas else 0
        recompensa_max = np.max(recompensas) if recompensas else 0
        recompensa_min = np.min(recompensas) if recompensas else 0

        # 🔹 Registro de métricas no TensorBoard
        with writer.as_default():
            tf.summary.scalar("Número de colisões", num_colisoes, step=episode)
            tf.summary.scalar("Tempo de sobrevivência (s)", tempo_sobrevivencia, step=episode)
            tf.summary.scalar("Distância percorrida (m)", distancia_percorrida, step=episode)
            tf.summary.scalar("Velocidade média (km/h)", velocidade_media, step=episode)
            tf.summary.scalar("Recompensa Média", recompensa_media, step=episode)
            tf.summary.scalar("Recompensa Máx", recompensa_max, step=episode)
            tf.summary.scalar("Recompensa Mín", recompensa_min, step=episode)
            writer.flush()

        print(f"✅ Episódio {episode} finalizado! Total reward: {total_reward}")

print("🏁 Treinamento concluído!")

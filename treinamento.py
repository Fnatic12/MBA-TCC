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
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

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
IM_WIDTH = 640
IM_HEIGHT = 480
REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 32
DISCOUNT = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 1000
FPS = 60
MODEL_NAME = "MobileNet_DQN"

# 🔹 Criando diretório para logs do TensorBoard
LOG_DIR = "logs_treinamento"
os.makedirs(LOG_DIR, exist_ok=True)
writer = tf.summary.create_file_writer(LOG_DIR)

# ✅ Ambiente CARLA
class CarEnv:
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.actor_list = []
        self.front_camera = None
        self.collision_hist = []

    def reset(self):
        """Reseta o ambiente e inicia um novo episódio"""
        self.clear_actors()
        self.collision_hist = []
        self.total_distance = 0  

        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, spawn_point)
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

        # 🔹 Sensor de colisão
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        return self.front_camera

    def process_img(self, image):
        """Processa a imagem da câmera"""
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        self.front_camera = i2[:, :, :3]

    def collision_data(self, event):
        """Detecta colisões"""
        self.collision_hist.append(event)

    def step(self, action):
        """Executa ação e retorna observações"""
        controls = [
            carla.VehicleControl(throttle=0.6, steer=-0.5),
            carla.VehicleControl(throttle=0.6, steer=0.0),
            carla.VehicleControl(throttle=0.6, steer=0.5),
            carla.VehicleControl(throttle=0.2, steer=0.0),
            carla.VehicleControl(throttle=0.0, brake=1.0)
        ]
        self.vehicle.apply_control(controls[action])

        time.sleep(1 / FPS)  

        v = self.vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)  

        self.total_distance += (kmh * 1000 / 3600) * (1 / FPS)  

        reward = kmh / 10  
        if len(self.collision_hist) != 0:
            reward = -200
            done = True
        elif action == 4:  
            reward -= 5
        else:
            done = False

        return self.front_camera, reward, done, kmh

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

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0

    def create_model(self):
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

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episódios'):
        env.collision_hist = []
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        total_velocity = 0
        rewards = []  

        while not done:
            if np.random.random() > EPSILON:
                action = np.argmax(agent.model.predict(np.expand_dims(state, axis=0) / 255.0))
            else:
                action = np.random.randint(0, 5)

            new_state, reward, done, kmh = env.step(action)
            total_reward += reward
            total_velocity += kmh
            rewards.append(reward)
            steps += 1

            agent.replay_memory.append((state, action, reward, new_state, done))
            state = new_state

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

        # 🔹 TREINAMENTO DA REDE NEURAL
        if len(agent.replay_memory) >= MIN_REPLAY_MEMORY_SIZE:
            minibatch = random.sample(agent.replay_memory, MINIBATCH_SIZE)
            states = np.array([transition[0] for transition in minibatch]) / 255.0
            actions = np.array([transition[1] for transition in minibatch])
            rewards = np.array([transition[2] for transition in minibatch])
            next_states = np.array([transition[3] for transition in minibatch]) / 255.0
            dones = np.array([transition[4] for transition in minibatch])

            current_qs = agent.model.predict(states, verbose=0)
            future_qs = agent.target_model.predict(next_states, verbose=0)

            for i in range(MINIBATCH_SIZE):
                target_q = rewards[i] if dones[i] else rewards[i] + DISCOUNT * np.max(future_qs[i])
                current_qs[i, actions[i]] = target_q

            loss_value = agent.model.train_on_batch(states, current_qs)

        # 🔹 Cálculo da velocidade média e distância percorrida
        avg_velocity = total_velocity / max(1, steps)  
        distance_covered = env.total_distance  

        # 🔹 REGISTRANDO MÉTRICAS NO TENSORBOARD
        with writer.as_default():
            tf.summary.scalar("Recompensa Total", total_reward, step=episode)
            tf.summary.scalar("Recompensa Média", np.mean(rewards), step=episode)
            tf.summary.scalar("Recompensa Máx", np.max(rewards), step=episode)
            tf.summary.scalar("Recompensa Mín", np.min(rewards), step=episode)
            tf.summary.scalar("Número de colisões", len(env.collision_hist), step=episode)
            tf.summary.scalar("Passos no Episódio", steps, step=episode)
            tf.summary.scalar("Tempo de sobrevivência (s)", steps / FPS, step=episode)
            tf.summary.scalar("Velocidade Média (km/h)", avg_velocity, step=episode)
            tf.summary.scalar("Distância Percorrida (m)", distance_covered, step=episode)
            tf.summary.scalar("Epsilon", EPSILON, step=episode)
            writer.flush()

        # 🔹 Salvando o modelo treinado após cada episódio
        agent.model.save("models/MobileNet_DQN.h5")
import glob
import os
import sys
import random
import time
import numpy as np
import math
import tensorflow as tf
from tqdm import tqdm

# Configuração do TensorFlow para GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Garantindo que o CARLA seja importado corretamente
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

# Configurações do ambiente
IM_WIDTH = 640
IM_HEIGHT = 480
EPISODES = 1000  
SPAWN_RETRIES = 10  
EPSILON = 1.0  # Mantendo fixo para baseline sem aprendizado
FPS = 60  # 🔹 Garantindo que o baseline use o mesmo FPS do treinamento

# Criando diretório de logs para o TensorBoard
LOG_DIR = "logs_baseline"
os.makedirs(LOG_DIR, exist_ok=True)
writer = tf.summary.create_file_writer(LOG_DIR)

# Classe para o ambiente no CARLA
class CarEnv:
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.actor_list = []
        self.front_camera = None
        self.error_count = 0  
        self.collision_hist = []

    def clear_actors(self):
        """Remove todos os veículos e sensores da simulação."""
        print("🧹 Limpando atores antigos...")
        for actor in self.actor_list:
            try:
                actor.destroy()
                time.sleep(0.5)
            except:
                pass
        self.actor_list = []
        time.sleep(1)  

    def process_image(self, image):
        """Processa a imagem capturada pela câmera do veículo."""
        img = np.array(image.raw_data)
        img = img.reshape((IM_HEIGHT, IM_WIDTH, 4))[:, :, :3]  # Removendo canal alfa
        img = np.transpose(img, (1, 0, 2))  # Ajustando a ordem de largura e altura
        self.front_camera = img

    def collision_data(self, event):
        """Registra colisões para encerrar episódios corretamente."""
        self.collision_hist.append(event)

    def reset(self):
        """Reseta o ambiente, spawnando um novo veículo e sensores."""
        self.clear_actors()
        self.collision_hist = []
        success = False

        for attempt in range(1, SPAWN_RETRIES + 1):
            try:
                spawn_point = random.choice(self.world.get_map().get_spawn_points())
                self.vehicle = self.world.spawn_actor(self.model_3, spawn_point)
                self.actor_list.append(self.vehicle)

                # Adiciona sensor de colisão
                colsensor = self.blueprint_library.find("sensor.other.collision")
                self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
                self.actor_list.append(self.colsensor)
                self.colsensor.listen(lambda event: self.collision_data(event))

                success = True
                break
            except:
                print(f"⚠️ Erro ao spawnar carro. Tentando novamente... ({attempt}/{SPAWN_RETRIES})")
                time.sleep(2)  

        if not success:
            self.error_count += 1  
            if self.error_count >= 3:
                print("🚨 Muitos erros seguidos! Reiniciando conexão com CARLA...")
                self.client = carla.Client("localhost", 2000)
                self.client.set_timeout(10.0)
                self.world = self.client.get_world()
                self.error_count = 0  

            raise RuntimeError("🚨 Falha ao spawnar o veículo após várias tentativas. Verifique o CARLA.")

    def step(self, action):
        """Executa a ação escolhida aleatoriamente e coleta métricas."""
        controls = [
            carla.VehicleControl(throttle=1.0, steer=-0.5),
            carla.VehicleControl(throttle=1.0, steer=0.0),
            carla.VehicleControl(throttle=1.0, steer=0.5),
            carla.VehicleControl(throttle=0.5, steer=0.0),
            carla.VehicleControl(throttle=0.0, brake=1.0)
        ]
        self.vehicle.apply_control(controls[action])

        time.sleep(1 / FPS)  # 🔹 Ajustando a simulação para manter o mesmo FPS

        v = self.vehicle.get_velocity()
        kmh = math.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6  
        return kmh

if __name__ == '__main__':
    env = CarEnv()

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episódios'):
        print(f"\n📌 Iniciando episódio {episode}/{EPISODES}")

        try:
            env.reset()
        except RuntimeError as e:
            print(e)
            continue  

        distancia_percorrida = 0
        velocidade_acumulada = 0
        num_colisoes = 0
        recompensas = []
        erros_direcao = 0

        done = False
        steps = 0  # 🔹 Contador de passos

        while not done:
            action = np.random.randint(0, 5)  # Escolhe um número de 0 a 4 com mesma probabilidade  # Ações aleatórias no baseline

            kmh = env.step(action)
            velocidade_acumulada += kmh
            steps += 1  # 🔹 Contagem de passos

            # 🚨 Agora detecta colisão corretamente e reinicia o episódio
            if len(env.collision_hist) > 0:
                num_colisoes += 1
                done = True
                
        velocidade_media = velocidade_acumulada / max(1, steps)  # 🔹 Ajustado para evitar divisão por zero
        tempo_sobrevivencia = steps / FPS  # 🔹 Agora consistente com o treinamento
        distancia_percorrida = velocidade_media * tempo_sobrevivencia  # 🔹 Ajustado para estar em METROS corretamente
        recompensa_media = np.mean(recompensas) if recompensas else 0
        recompensa_max = np.max(recompensas) if recompensas else 0
        recompensa_min = np.min(recompensas) if recompensas else 0
        erros_direcao = num_colisoes  

        # Registro de métricas no TensorBoard
        with writer.as_default():
            tf.summary.scalar("Número de colisões", num_colisoes, step=episode)
            tf.summary.scalar("Tempo de sobrevivência (s)", tempo_sobrevivencia, step=episode)
            tf.summary.scalar("Distância percorrida (m)", distancia_percorrida, step=episode)  # 🔹 AGORA EM METROS
            tf.summary.scalar("Velocidade média (km/h)", velocidade_media, step=episode)
            tf.summary.scalar("Recompensa Média", recompensa_media, step=episode)
            tf.summary.scalar("Recompensa Máx", recompensa_max, step=episode)
            tf.summary.scalar("Recompensa Mín", recompensa_min, step=episode)
            writer.flush()

    print("\n🏁 Baseline concluído!")

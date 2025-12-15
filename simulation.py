import math
import random
import sys
import os
import pickle
import neat
import pygame

WIDTH = 1366
HEIGHT = 768

BORDER_COLOR = (255, 255, 255, 255) 

current_generation = 0 

class Coord:
    def __init__(self, x, y):
        self._x = x
        self._y = y

s_coord = {
    "bah.png": Coord(806, 664),
    "bel.png": Coord(491, 325),
    "ita.png": Coord(908, 552),
    "gbr.png": Coord(902, 451),
    "jpn.png": Coord(599, 614),
    "rand.png": Coord(776, 604),
    "rand_hard.png": Coord(876, 647)
}

track_name = "rand_hard.png"

class Car:
    def __init__(self):
        self.sprite = pygame.image.load('assets/w10_min.png').convert_alpha() # Convert Speeds Up A Lot
        self.rotated_sprite = self.sprite 
        coord = s_coord[track_name] 
        self.angle = 0
        self.speed = 0
        self.position = [coord._x, coord._y]
        self.speed_set = False 
        self.center = [self.position[0] + 26, self.position[1] + 14.5] 
        self.sensors = [] 
        self.alive = True 
        self.distance = 0 
        self.f = 0
        self.time = 0 

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position) 
        #self.draw_sensor(screen) 

    def draw_sensor(self, screen):
        for sensor in self.sensors:
            position = sensor[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_sensor(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))))
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))))
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 150:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.sensors.append([(x, y), dist])
    
    def update(self, game_map):
        if not self.speed_set:
            self.speed = 9
            self.speed_set = True

        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] -= math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 8)
        self.position[0] = min(self.position[0], WIDTH - 29)

        self.distance += self.speed
        self.time += 1
        
        self.position[1] -= math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 8)
        self.position[1] = min(self.position[1], WIDTH - 52)

        self.center = [int(self.position[0]) + 26, int(self.position[1]) + 14.5]

        length = 0.5 * 52
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length - 2, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length] 
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length - 2, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length] 
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]
        self.check_collision(game_map)
        self.sensors.clear()

        for d in range(-270, -45, 45):
            self.check_sensor(d, game_map)

    def get_data(self):
        sensors = self.sensors
        return_values = [0, 0, 0, 0, 0]
        for i, sensor in enumerate(sensors):
            return_values[i] = int(sensor[1] / 30)
        return return_values

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / 26

    def rotate_center(self, image, angle):
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rect = rotated_image.get_rect(center=image.get_rect(topleft=self.position).center)
        return rotated_image
    
def run_simulation(genomes, config):
    nets = []
    cars = []
    tolerance = 10

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load('assets/' + track_name).convert() 

    global current_generation
    current_generation += 1
    counter = 0
    terminate_simulation = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                if car.speed > 4:
                    car.angle += 2
                else:
                    car.angle += 4
            elif choice == 1:
                if car.speed > 4:
                    car.angle -= 2
                else:
                    car.angle -= 4
            elif choice == 2:
                if (car.speed > 2.5):
                    car.speed -= 0.1
            else:
                if car.speed >= 3 and car.speed < 6: 
                    car.speed += 0.09
                elif car.speed < 3:
                    car.speed += 0.19
                
        still_alive = 0
        
        global f
        for i, car in enumerate(cars):
            f = 0
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                dist_reward = car.get_reward()
                time_penalty = car.time / 60
                speed_penalty = max(0, 3 - car.speed) * 10
                genomes[i][1].fitness += (dist_reward - time_penalty - speed_penalty)
                car_x, car_y = car.position[0], car.position[1]
                coord = s_coord[track_name]
                distance = math.sqrt((car_x - coord._x) ** 2 + (car_y - coord._y) ** 2)
                if car.distance > 100 and distance <= tolerance: 
                    genomes[i][1].fitness += 1000
                    terminate_simulation = True
                
        if terminate_simulation:
            print("Goal achieved")
            break

        if still_alive == 0:
            break

        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        pygame.display.flip()
        clock.tick(60) 

if __name__ == "__main__":
    config_path = "neat-config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    try:
        t_name = track_name[:len(track_name)-4]
        with open(t_name + '_neat_population.pkl', 'rb') as f:
            population = pickle.load(f)
        print("Loaded population from file")
    except:
        print("Starting simulation from scratch...")
        population = neat.Population(config)
        
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    population.run(run_simulation, 150)
    with open(t_name + '_neat_population.pkl', 'wb') as f:
        pickle.dump(population, f)

import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm

# Load CSV file data
data = pd.read_csv("India Cities LatLng.csv", encoding='latin-1')

# Calculate distance matrix
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the distance between two latitude and longitude coordinates (unit: kilometers)"""
    R = 6371  # Earth radius (kilometers)
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

num_cities = len(data)
distance_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        distance = haversine(data.lat[i], data.lng[i], data.lat[j], data.lng[j])
        distance_matrix[i][j] = distance

# Assume that the time cost matrix is the same as the distance matrix
time_matrix = distance_matrix

# PSO parameter settings
num_particles = 20
num_iterations = 100
w = 0.8  # Inertia weight
c1 = 1.5  # Individual learning factor
c2 = 1.5  # Social learning factor

# Particle class
class Particle:
    def __init__(self):
        self.position = random.sample(range(num_cities), num_cities)
        self.velocity = [0] * num_cities
        self.best_position = self.position.copy()
        self.best_distance = self.calculate_distance()
        self.best_time = self.calculate_time()

    def calculate_distance(self):
        total_distance = 0
        for i in range(num_cities - 1):
            total_distance += distance_matrix[self.position[i]][
                self.position[i + 1]
            ]
        total_distance += distance_matrix[self.position[-1]][self.position[0]]
        return total_distance

    def calculate_time(self):
        total_time = 0
        for i in range(num_cities - 1):
            total_time += time_matrix[self.position[i]][self.position[i + 1]]
        total_time += time_matrix[self.position[-1]][self.position[0]]
        return total_time

    def update_velocity(self, global_best_position):
        for i in range(num_cities):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = (
                c1 * r1 * (self.best_position[i] - self.position[i])
            )
            social_velocity = (
                c2 * r2 * (global_best_position[i] - self.position[i])
            )
            self.velocity[i] = int(
                w * self.velocity[i] + cognitive_velocity + social_velocity
            )

    def update_position(self):
        # Update path using swap operation
        for i in range(num_cities):
            j = int(i + self.velocity[i]) % num_cities
            self.position[i], self.position[j] = (
                self.position[j],
                self.position[i],
            )
        self.best_distance = self.calculate_distance()
        self.best_time = self.calculate_time()


# Initialize the particle swarm
particles = [Particle() for _ in range(num_particles)]

# Initialize the global best path
global_best_position = particles[0].position.copy()
global_best_distance = particles[0].best_distance
global_best_time = particles[0].best_time

# Iterative search
for _ in range(num_iterations):
    for particle in particles:
        particle.update_velocity(global_best_position)
        particle.update_position()
        if particle.best_distance < global_best_distance:
            global_best_position = particle.position.copy()
            global_best_distance = particle.best_distance
        if particle.best_time < global_best_time:
            global_best_time = particle.best_time

print("Shortest path:", global_best_position)
print("Shortest path length:", global_best_distance)
print("Time saving path:", global_best_position)  # In this example, the time cost is the same as the distance
print("Time saving path time:", global_best_time)

# Draw heatmap and path map
plt.figure(figsize=(15, 8))

# Subplot 1: City distribution
plt.subplot(2, 2, 1)
plt.scatter(data.lng, data.lat, c='lightblue', s=50, label="Cities")
plt.scatter(data.lng[global_best_position[0]], data.lat[global_best_position[0]], c='red', s=100, label="Start point")
plt.scatter(data.lng[global_best_position[-1]], data.lat[global_best_position[-1]], c='green', s=100, label="End point")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("City Distribution Map")
plt.legend()

# Subplot 2: Shortest path
plt.subplot(2, 2, 2)
plt.scatter(data.lng, data.lat, c='lightblue', s=50, label="Cities")
plt.plot([data.lng[i] for i in global_best_position], [data.lat[i] for i in global_best_position], 'r-', linewidth=1, label="Shortest path")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Shortest Path Map")
plt.legend()

# Subplot 3: Time saving path
plt.subplot(2, 2, 3)
plt.scatter(data.lng, data.lat, c='lightblue', s=50, label="Cities")
plt.plot([data.lng[i] for i in global_best_position], [data.lat[i] for i in global_best_position], 'g-', linewidth=1, label="Time saving path")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Time Saving Path Map")
plt.legend()

# Subplot 4: Distance heatmap
plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
ax = plt.gca()  # 获取当前子图对象
plt.scatter(data.lng, data.lat, transform=ccrs.PlateCarree(), c='lightblue', s=50, label="Cities")

# Add map background
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.LAND, color='lightgreen')
ax.add_feature(cfeature.OCEAN, color='lightblue')

# Draw distance heatmap
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            plt.plot([data.lng[i], data.lng[j]], [data.lat[i], data.lat[j]], color=cm.plasma(distance_matrix[i, j] / np.max(distance_matrix)), linewidth=1, transform=ccrs.PlateCarree())

# Add title
plt.title("Distance Heatmap")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()  # Adjust subplot spacing
plt.show()

import math
import numpy as np
from PIL import Image

from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import sharedctypes


class Triangle:
    def __init__(self, vertices, normal)
        pass

    def intersect(self, ray):
        a = self.vertices[0].x-self.vertices[1].x
        b = self.vertices[0].y-self.vertices[1].y
        c = self.vertices[0].z-self.vertices[1].z

        d = self.vertices[0].x-self.vertices[2].x
        e = self.vertices[0].y-self.vertices[2].y
        f = self.vertices[0].z-self.vertices[2].z

        g = ray.direction[0]
        h = ray.direction[1]
        i = ray.direction[2]


class Sphere:
    def __init__(self, center, radius, color, specular=-1, reflective=0):
        self.center = center
        self.radius = radius
        self.color = color
        self.specular = specular
        self.reflective = reflective
        
        # caches
        self.CO = None
        self.rsq = self.radius * self.radius
    
    def intersect(self, a, ray):
        if self.CO is None:
            self.CO = ray.origin - self.center
            self.COdotCO = np.dot(self.CO, self.CO)
            
        b = 2 * np.dot(self.CO, ray.direction)
        c = self.COdotCO - self.rsq

        disc = b*b - 4*a*c
        
        if disc < 0:
            return None, None
        else:
            s = math.sqrt(disc)
            t1 = (-b + s)/(2*a)
            t2 = (-b - s)/(2*a)

            return t1, t2
    
    def get_normal(self, point):
        normal = normalize(point - self.center)
        return normal


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction


class Scene:
    def __init__(self, objects, light_sources):
        self.objects = objects
        self.light_sources = light_sources
        self.hit_lookup = {}


class Lightning:
    def __init__(self, intensity):
        self.intensity = intensity


class PointLightning(Lightning):
    def __init__(self, position, intensity):
        super(PointLightning, self).__init__(intensity)
        self.position = position


class AmbientLightning(Lightning):
    def __init__(self, intensity):
        super(AmbientLightning, self).__init__(intensity)


class DirectionalLightning(Lightning):
    def __init__(self, intensity, direction):
        super(DirectionalLightning, self).__init__(intensity)
        self.direction = direction


def normalize(vector):
    return vector/np.linalg.norm(vector, ord=2)


def lambert_shader(scene, ray, P, obj):
    intensity = 0.0
    N = obj.get_normal(P)

    for light_source in scene.light_sources:
        if isinstance(light_source, AmbientLightning):
            intensity += light_source.intensity
        else:
            if isinstance(light_source, PointLightning):
                L = light_source.position - P
            else:
                L = light_source.direction

            dot_product = np.dot(N, normalize(L))
            intensity += light_source.intensity * max(0, dot_product)

    return intensity * obj.color


def phong_shader(scene, ray, P, obj):
    intensity = 0.0
    N = obj.get_normal(P)

    for light_source in scene.light_sources:
        if isinstance(light_source, AmbientLightning):
            intensity += light_source.intensity
        else:
            if isinstance(light_source, PointLightning):
                L = light_source.position - P
            else:
                L = light_source.direction
            
            dot_product = np.dot(N, normalize(L))
            intensity += light_source.intensity * max(0, dot_product)

            if obj.specular != -1:
                V = -ray.direction
                H = normalize(2 * N * np.dot(N, L) - L)
                S = np.dot(normalize(V), H)
                intensity += light_source.intensity * max(0, S)**obj.specular

    intensity = min(intensity, 1.0)
    return intensity * obj.color


def phong_shadow_shader(scene, ray, P, obj):
    intensity = 0.0
    N = obj.get_normal(P)

    for light_source in scene.light_sources:
        if isinstance(light_source, AmbientLightning):
            intensity += light_source.intensity
        else:
            # shadow
            # TODO: add shadow coherence
            if isinstance(light_source, PointLightning):
                L = light_source.position - P
                t_max = 1
            else:
                L = light_source.direction
                t_max = math.inf

            if intersect_any(scene, P, L, 0.001, t_max):
                continue

            # light
            dot_product = np.dot(N, normalize(L))
            intensity += light_source.intensity * max(0, dot_product)

            if obj.specular != -1:
                V = -ray.direction
                H = normalize(2 * N * np.dot(N, L) - L)
                S = np.dot(normalize(V), H)
                intensity += light_source.intensity * max(0, S)**obj.specular

    intensity = min(intensity, 1.0)
    return intensity * obj.color


def save_image(image_buffer, write_file):
    im = Image.fromarray(image_buffer)
    im.save(write_file)


def get_x_rotation(angle):
    return np.array([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]])


def get_y_rotation(angle):
    return np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])


def get_z_rotation(angle):
    return np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])


def render_scene(scene, resolution, out_file):
    width, height = resolution
    rendering = np.ctypeslib.as_ctypes(np.zeros((height, width, 3)))
    shared_array = sharedctypes.RawArray(rendering._type_, rendering)

    d = 1
    viewport_w, viewport_h = (d, d)
    camera = np.array([0, 0, 0])
    # camera = np.array([0, 5, 0])

    rotation = np.eye(3)
    # rotation = get_z_rotation(math.radians(10))
    # rotation = rotation @ get_x_rotation(math.radians(20))

    # shader = lambert_shader
    # shader = phong_shader
    shader = phong_shadow_shader

    pixels = [(x, y) for x in range(-width//2, width//2) for y in range(-height//2, height//2)]
    rate = 2 
    
    def render(pixels):
        tmp = np.ctypeslib.as_array(shared_array)

        for x, y in pixels:
            if x % rate == 0 and y % rate == 0:
                world_coords = rotation @ canvas_to_viewport(x, y, d, viewport_w, viewport_h, width, height)
                pixel = trace_ray(shader, camera, x, y, world_coords, scene)
                tmp[height-1-(y+height//2), x+width//2, :] = pixel

        # for x, y in pixels:
        #     if x % rate != 0 or y % rate != 0:
        #         world_coords = rotation @ canvas_to_viewport(x, y, d, viewport_w, viewport_h, width, height)
        #         pixel = lookup_ray(shader, camera, x, y, world_coords, scene)
        #         tmp[height-1-(y+height//2), x+width//2, :] = pixel

    num_processes = 4
    chunk_size = len(pixels) // num_processes
    divisions = [pixels[i*chunk_size:(i+1)*chunk_size] for i in range(num_processes)]
    processes = [Process(target=render, args=[divisions[i]]) for i in range(num_processes)]
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    save_image(np.ctypeslib.as_array(shared_array).astype(np.uint8), out_file)


def intersect_ray(scene, origin, direction, t_min=1, t_max=math.inf):
    ray = Ray(origin=origin, direction=direction)
    
    closest_distance = math.inf
    closest_object = None

    a = np.dot(ray.direction, ray.direction)

    for obj in scene.objects:
        t1, t2 = obj.intersect(a, ray)

        if t1 is not None and t1 < closest_distance and t_max >= t1 >= t_min:
            closest_distance = t1
            closest_object = obj

        if t2 is not None and t2 < closest_distance and t_max >= t2 >= t_min:
            closest_distance = t2
            closest_object = obj
    
    return closest_object, closest_distance, ray


def intersect_any(scene, origin, direction, t_min=1, t_max=math.inf):
    ray = Ray(origin=origin, direction=direction)

    a = np.dot(ray.direction, ray.direction)

    for obj in scene.objects:
        t1, t2 = obj.intersect(a, ray)

        if t1 is not None and t_max >= t1 >= t_min:
            return True

        if t2 is not None and t_max >= t2 >= t_min:
            return True
    
    return False


def intersect_object(obj, origin, direction, t_min=1, t_max=math.inf):
    ray = Ray(origin=origin, direction=direction)

    a = np.dot(ray.direction, ray.direction)

    t1, t2 = obj.intersect(a, ray)

    if t1 is not None and t_max >= t1 >= t_min:
        intersection = ray.origin + t1 * ray.direction
        return ray, intersection

    if t2 is not None and t_max >= t2 >= t_min:
        intersection = ray.origin + t2 * ray.direction
        return ray, intersection
    
    return None


def lookup_ray(shader, camera, x, y, world_coords, scene):
    max_dist = 10

    for i in range(1, 10):
        for n in range(-i, i):
            f, g = x+n, y
                
            if (f, g) in scene.hit_lookup:
                obj = scene.hit_lookup[(f, g)]
                a = intersect_object(obj, camera, world_coords)
                if a is not None:
                   ray, intersection = a
                   return shader(scene, ray, intersection, obj)
            else:
                trace_ray(shader, camera, x, y, world_coords, scene)
    

def trace_ray(shader, camera, x, y, world_coords, scene):
    closest_object, closest_distance, ray = intersect_ray(scene, camera, world_coords)

    if closest_object is None:
        return np.array([255, 255, 255])

    intersection = ray.origin + closest_distance * ray.direction
    local_color = shader(scene, ray, intersection, closest_object)

    r = closest_object.reflective
    if closest_object.reflective <= 0:
        scene.hit_lookup[(x, y)] = closest_object
        return local_color
    else:
        for i in range(2):
            reflection = reflect_vector(-ray.direction, closest_object.get_normal(intersection))
            object_i, dist_i, ray_i = intersect_ray(scene, intersection, reflection, t_min=0.001)

            if object_i is None:
                break
            else:
                intersection = ray_i.origin + dist_i * ray_i.direction
                closest_object = object_i
                ray = ray_i

        color = shader(scene, ray, intersection, closest_object)
        final_color = local_color * (1-r) + color * r

        return final_color


def reflect_vector(vector, normal):
    return 2 * normal * np.dot(normal, vector) - vector


def canvas_to_viewport(x, y, dist_to_viewport, vw, vh, cw, ch):
    vx = x * (vw/cw)
    vy = y * (vh/ch)

    return np.array([vx, vy, dist_to_viewport])
    

if __name__ == '__main__':
    lightning = [
                 AmbientLightning(intensity=0.2),
                 PointLightning(intensity=0.6, position=np.array([2, 1, 0])),
                 DirectionalLightning(intensity=0.2, direction=np.array([1, 4, 4])),
                ]

    objects = [
               Sphere(center=np.array([0, -1, 3]), 
                      radius=1, 
                      color=np.array([255, 0, 0]),
                      specular=500,
                      reflective=0.0),
               Sphere(center=np.array([2, 0, 4]), 
                      radius=1, 
                      color=np.array([0, 0, 255]),
                      specular=500, reflective=0.0),
               Sphere(center=np.array([-2, 0, 4]), 
                      radius=1, 
                      color=np.array([0, 255, 0]),
                      specular=10),
               Sphere(center=np.array([0, -5001, 0]), 
                      radius=5000, 
                      color=np.array([255, 255, 0]),
                      specular=1000),
    ]

    scene = Scene(objects, lightning)
    render_scene(scene, (600, 600), 'result.png')

import cv2 as cv
import pybullet as p


class DomainRandomizer:
	def __init__(self, np_random):
		self.np_random = np_random

	def randomize_lighting(self, *imgs):
		"""
		Function for domain randomization of environment lighting
		:param imgs: images with rgb channels only
		:return modified_imgs: rgb images with modified colors and brightness
		"""
		modified_imgs = []
		for img in imgs:
			# Adjust brightness and contrast (beta, alpha)
			# 		alpha 1  beta 0      --> no change
			# 		0 < alpha < 1        --> lower contrast
			# 		alpha > 1            --> higher contrast
			# 		-127 < beta < +127   --> good range for brightness values

			contrast = 1
			brightness = self.np_random.randint(-80, 60)
			img = cv.addWeighted(img, contrast, img, 0, brightness)

			# Adjust hue, saturation and lightness
			# 		hue range			--> [0, 255]
			# 		saturation range	--> [0, 255]
			# 		lightness range		--> [0, 255]
			hue = self.np_random.randint(0, 256)
			saturation = self.np_random.randint(0, 30)
			lightness = 0
			hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
			h, s, v = cv.split(hsv)
			h += hue
			s += saturation
			v += lightness

			# Merge channels
			img = cv.merge((h, s, v))
			img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
			modified_imgs.append(img)

		return modified_imgs

	def randomize_physics_constraints(self, body_id, link=-1, mass_range=(0.5, 4), friction_range=(20, 200)):
		"""
		Randomizes mass and friction of a body or link
		:param body_id: Unique body id
		:param link: -1 is base link
		:param mass_range: random number in this range
		:param friction_range: random number in this range
		"""
		# Randomize mass of object and lateral friction
		mass = self.np_random.uniform(*mass_range)
		friction = self.np_random.uniform(*friction_range)

		p.changeDynamics(body_id, link, mass=mass, lateralFriction=friction)






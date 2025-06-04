import folder_paths
import torch
import os
import sys

import cv2
import glob
import time
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw

this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_path)

import gpeno.__init_paths
from gpeno.face_enhancement import FaceEnhancement

models_dir = folder_paths.models_dir

global_gpen_processor = None
global_gpen_cache_model = ""


class GPENO:
	"""Restores faces in the image a special version of the GPEN technique that has been optimized for speed."""

	@classmethod
	def INPUT_TYPES(cls):
		return {
		    "required": {
		        "image": ("IMAGE", ),
		        "use_global_cache": ("BOOLEAN", {
		            "default": True,
		            "tooltip": "If enabled, the model will be loaded once and shared across all instances of this node. This saves VRAM if you are using multiple instances of GPENO in your flow, but the settings must remain the same for all instances."
		        }),
		        "unload": ("BOOLEAN", {
		            "default": False,
		            "tooltip": "If enabled, the model will be freed from the cache at the start of this node's execution (if applicable), and it will not be cached again."
		        }),
		        "backbone": (["RetinaFace-R50", "mobilenet0.25_Final"], {
		            "default": "RetinaFace-R50",
		            "tooltip": "Backbone files are downloaded to `comfyui/models/facedetection`."
		        }),
		        "resolution_preset": (["512", "1024", "2048"], {
		            "default": "512"
		        }),
		        "downscale_method": (["Bilinear", "Nearest", "Bicubic", "Area", "Lanczos"], {
		            "default": "Bilinear"
		        }),
		        "channel_multiplier": ("FLOAT", {
		            "default": 2
		        }),
		        "narrow": ("FLOAT", {
		            "default": 1.0
		        }),
		        "alpha": ("FLOAT", {
		            "default": 1.0
		        }),
		        "device": (["cpu", "cuda"], {
		            "default": "cuda" if torch.cuda.is_available() else "cpu"
		        }),
		        "aligned": ("BOOLEAN", {
		            "default": False
		        }),
		        "colorize": ("BOOLEAN", {
		            "default": False,
		        }),
		    },
		}

	RETURN_TYPES = (
	    "IMAGE",
	    "IMAGE",
	    "IMAGE",
	)
	FUNCTION = "op"
	CATEGORY = "image"
	DESCRIPTION = """
Performs GPEN face restoration on the input image(s). This implementation has been optimized for speed.
"""

	def __init__(self):
		self.gpen_processor = None
		self.gpen_cache_model = ""

	def op(self, image, use_global_cache, unload, backbone, resolution_preset, downscale_method, channel_multiplier, narrow, alpha, device, aligned, colorize):
		global global_gpen_processor
		global global_gpen_cache_model

		# Package arguments into attribute notation for use with argparse
		args = argparse.Namespace()
		args.model = f"GPEN-BFR-{resolution_preset}"
		args.channel_multiplier = channel_multiplier
		args.narrow = narrow
		args.alpha = alpha
		args.use_cuda = device
		args.aligned = aligned

		# Hardcoded arguments irrelevant to the user
		args.use_sr = False
		args.in_size = int(resolution_preset)
		args.out_size = 0
		args.sr_model = "realesrnet"
		args.sr_scale = 2
		args.key = None
		args.indir = "example/imgs"
		args.outdir = "results/outs-BFR"
		args.ext = ".jpg"
		args.save_face = False
		args.tile_size = 0

		if downscale_method == "Nearest":
			downscale_method = cv2.INTER_NEAREST
		elif downscale_method == "Bilinear":
			downscale_method = cv2.INTER_LINEAR
		elif downscale_method == "Area":
			downscale_method = cv2.INTER_AREA
		elif downscale_method == "Cubic":
			downscale_method = cv2.INTER_CUBIC
		elif downscale_method == "Lanczos":
			downscale_method = cv2.INTER_LANCZOS4

		def download_file(filename, url, logger=None, overwrite=False, headers=None):
			import os, requests

			# log = get_logger(logger)

			if overwrite or not os.path.exists(filename):
				# Make sure directory structure exists
				os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

				# log.info(f"Downloading file into: {filename}...")
				if os.path.exists("/stable-diffusion-cache/models/facedetection"):
					global models_dir
					models_dir = "/stable-diffusion-cache/models"
				else:
					response = requests.get(url, stream=True, headers=headers)
					if response.status_code != 200:
						# log.error(f"Error when trying to download `{url}` to `{filename}`. Dtatus code received: {response.status_code}")
						return False
					try:
						with open(filename, "wb") as fout:
							for block in response.iter_content(4096):
								fout.write(block)
					except:
						# log.exception(f"Error when writing download to `{filename}`.")
						return False

			return True

		gpen_dir = os.path.join(models_dir, "facerestore_models")
		facedetect_dir = os.path.join(models_dir, "facedetection")

		if args.model == "GPEN-BFR-512":
			download_file(f"{gpen_dir}/{args.model}.pth", "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth")
		elif args.model == "GPEN-BFR-1024":
			if not download_file(f"{gpen_dir}/{args.model}.pth", "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-1024.pth"):
				pass
				# self.log.error("The download link for the 1024 model doesn't appear to work. Try installing it manually into your unprompted/models/gpen folder: https://cyberfile.me/644d")
		elif args.model == "GPEN-BFR-2048":
			download_file(f"{gpen_dir}/{args.model}.pth", "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-2048.pth")

		if colorize:
			download_file(f"{gpen_dir}/GPEN-Colorization-1024.pth", "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-Colorization-1024.pth")

		# Additional dependencies
		download_file(f"{facedetect_dir}/parsing_parsenet.pth", "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/ParseNet-latest.pth")
		# for sr
		# download_file(f"{facedetect_dir}/realesrnet_x2.pth", "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/realesrnet_x2.pth")

		if backbone == "RetinaFace-R50":
			backbone = "detection_Resnet50_Final"
			download_file(f"{facedetect_dir}/detection_Resnet50_Final.pth", "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth")
		elif backbone == "mobilenet0.25_Final":
			backbone = "detection_mobilenet0.25_Final"
			download_file(f"{facedetect_dir}/detection_mobilenet0.25_Final.pth", "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth")

		if use_global_cache:
			this_gpen_processor = global_gpen_processor
			this_gpen_cache_model = global_gpen_cache_model
		else:
			this_gpen_processor = self.gpen_processor
			this_gpen_cache_model = self.gpen_cache_model

		if unload or not this_gpen_processor or this_gpen_cache_model != resolution_preset:
			print("Loading FaceEnhancement object...")
			face_enhancement_obj = FaceEnhancement(args, base_dir=f"{models_dir}/", in_size=args.in_size, model=args.model, use_sr=args.use_sr, device=args.use_cuda, interp=downscale_method, backbone=backbone, colorize=colorize)

			if use_global_cache:
				global_gpen_processor = face_enhancement_obj
				global_gpen_cache_model = resolution_preset
				this_gpen_processor = global_gpen_processor
				this_gpen_cache_model = global_gpen_cache_model
			else:
				self.gpen_processor = face_enhancement_obj
				self.gpen_cache_model = resolution_preset
				this_gpen_processor = self.gpen_processor
				this_gpen_cache_model = self.gpen_cache_model
		else:
			print("Using cached FaceEnhancement object.")
			# self.log.info("Using cached FaceEnhancement object.")

		print("Starting GPENO processing...")

		total_images = image.shape[0]
		out_images = []
		out_original_faces = []
		out_enhanced_faces = []

		for i in range(total_images):
			# image is a 4d tensor array in the format of [B, H, W, C]
			this_img = 255. * image[i].cpu().numpy()
			img = np.clip(this_img, 0, 255).astype(np.uint8)

			result, orig_faces, enhanced_faces = this_gpen_processor.process(img, aligned=args.aligned)

			out_images.append(result)
			# add each of the orig_faces list to the out_original_faces list
			for orig_face in orig_faces:
				out_original_faces.append(orig_face)
			for enhanced_face in enhanced_faces:
				out_enhanced_faces.append(enhanced_face)

		restored_img_np = np.array(out_images).astype(np.float32) / 255.0
		restored_img_tensor = torch.from_numpy(restored_img_np)

		restored_original_faces_np = np.array(out_original_faces).astype(np.float32) / 255.0
		restored_original_faces_tensor = torch.from_numpy(restored_original_faces_np)

		restored_enhanced_faces_np = np.array(out_enhanced_faces).astype(np.float32) / 255.0
		restored_enhanced_faces_tensor = torch.from_numpy(restored_enhanced_faces_np)

		if unload:
			print("Unloading GPEN from cache.")

			if use_global_cache:
				global_gpen_processor = None
				global_gpen_cache_model = ""
			else:
				self.gpen_cache_model = ""
				self.gpen_processor = None

		print("GPENO processing done.")
		return (
		    restored_img_tensor,
		    restored_original_faces_tensor,
		    restored_enhanced_faces_tensor,
		)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "GPENO Face Restoration": GPENO,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "GPENO Face Restoration": "GPENO Face Restoration",
}

import cv2
import numpy as np
import torch
from gpeno.face_detect.retinaface_detection import RetinaFaceDetection
# from gpeno.face_detect.batch_face import RetinaFace
from gpeno.face_parse.face_parsing import FaceParse
from gpeno.face_model.face_gan import FaceGAN
# from gpen.sr_model.real_esrnet import RealESRNet
from gpeno.align_faces import warp_and_crop_face, get_reference_facial_points


class FaceEnhancement(object):

	def __init__(self, args, base_dir='./', in_size=512, out_size=None, model=None, use_sr=True, device='cuda', interp=3, backbone="RetinaFace-R50", log=None, colorize=False):
		self.log = log
		# self.log.debug("Initializing FaceEnhancement...")
		self.facedetector = RetinaFaceDetection(base_dir, device, network=backbone)
		# self.facedetector = RetinaFace()
		self.facegan = FaceGAN(base_dir, in_size, out_size, model, args.channel_multiplier, args.narrow, args.key, device=device)
		# self.srmodel = RealESRNet(base_dir, args.sr_model, args.sr_scale, args.tile_size, device=device)
		self.faceparser = FaceParse(base_dir, device=device)
		self.use_sr = use_sr
		self.in_size = in_size
		self.out_size = in_size if out_size is None else out_size
		self.threshold = 0.9
		self.alpha = args.alpha
		self.interp = interp
		self.colorize = colorize

		if self.colorize:
			self.colorizer = FaceGAN(base_dir, 1024, 1024, "GPEN-Colorization-1024", args.channel_multiplier, args.narrow, None, device)

		self.mask = np.zeros((512, 512), np.float32)
		cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
		self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)

		self.kernel = np.array(([0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]), dtype="float32")
		self.reference_5pts = get_reference_facial_points((self.in_size, self.in_size), 0.25, (0, 0), True)

	def mask_postprocess(self, mask, thres=26):
		mask[:thres, :] = 0
		mask[-thres:, :] = 0
		mask[:, :thres] = 0
		mask[:, -thres:] = 0
		mask = cv2.GaussianBlur(mask, (101, 101), 4)
		return mask.astype(np.float32)

	def colorize_face(self, face):
		# Convert BGR (OpenCV) to RGB before colorizing
		rgb_input = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		color_face = self.colorizer.process(rgb_input)
		color_face = cv2.cvtColor(color_face, cv2.COLOR_RGB2BGR)

		if face.shape[:2] != color_face.shape[:2]:
			out_rs = cv2.resize(color_face, face.shape[:2][::-1])
			gray_yuv = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
			out_yuv = cv2.cvtColor(out_rs, cv2.COLOR_BGR2YUV)

			out_yuv[:, :, 0] = gray_yuv[:, :, 0]
			color_face = cv2.cvtColor(out_yuv, cv2.COLOR_YUV2BGR)

		return color_face

	def process(self, img, aligned=False):
		orig_faces, enhanced_faces = [], []
		if aligned:
			print("Aligned is true")
			ef = self.facegan.process(img)
			if self.colorize:
				ef = self.colorize_face(ef)
			orig_faces.append(img)
			enhanced_faces.append(ef)

			# if self.use_sr:
			#	ef = self.srmodel.process(ef)

			return ef, orig_faces, enhanced_faces

		# if self.use_sr:
		# 	img_sr = self.srmodel.process(img)
		#	if img_sr is not None:
		#		img = cv2.resize(img, img_sr.shape[:2][::-1])

		with torch.no_grad():
			print("Starting face detection")
			facebs, landms = self.facedetector.detect(img)
		# faces = self.facedetector.detect(img)

		# self.log.debug("Face detection complete")
		print("Face detection complete")

		height, width = img.shape[:2]
		full_mask = np.zeros((height, width), dtype=np.float32)
		full_img = np.zeros(img.shape, dtype=np.uint8)

		# for i, (faceb, facial5points, score) in enumerate(faces):
		for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
			if faceb[4] < self.threshold:
				continue
			fh, fw = (faceb[3] - faceb[1]), (faceb[2] - faceb[0])

			facial5points = np.reshape(facial5points, (2, 5))

			print("Starting face alignment...")
			of, tfm_inv = warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts, crop_size=(self.in_size, self.in_size))

			print("Face alignment complete")

			# enhance the face
			ef = self.facegan.process(of)

			if self.colorize:
				ef = self.colorize_face(ef)

			# self.log.debug("Face GAN complete")
			print("Face GAN complete")

			orig_faces.append(of)
			enhanced_faces.append(ef)

			tmp_mask = self.mask
			tmp_mask = self.mask_postprocess(self.faceparser.process(ef)[0] / 255.)
			# self.log.debug("Mask postprocessing complete")
			tmp_mask = cv2.resize(tmp_mask, (self.in_size, self.in_size), interpolation=self.interp)

			tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=self.interp)

			# from PIL import Image

			# tmp_mask_pil = Image.fromarray(tmp_mask)
			# Apply the inverse affine transformation using PIL
			# tmp_mask_pil = tmp_mask_pil.transform((width, height), Image.AFFINE, tuple(tfm_inv.flatten()), resample=self.interp)
			# tmp_mask = np.array(tmp_mask_pil)

			if min(fh, fw) < 100:  # gaussian filter for small faces
				ef = cv2.filter2D(ef, -1, self.kernel)

			ef = cv2.addWeighted(ef, self.alpha, of, 1. - self.alpha, 0.0)

			if self.in_size != self.out_size:
				ef = cv2.resize(ef, (self.in_size, self.in_size), interpolation=self.interp)
			tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=self.interp)

			mask = tmp_mask - full_mask
			full_mask[np.where(mask > 0)] = tmp_mask[np.where(mask > 0)]
			full_img[np.where(mask > 0)] = tmp_img[np.where(mask > 0)]

		full_mask = full_mask[:, :, np.newaxis]
		# if self.use_sr and img_sr is not None:
		# 	img = cv2.convertScaleAbs(img_sr * (1 - full_mask) + full_img * full_mask)
		# else:
		img = cv2.convertScaleAbs(img * (1 - full_mask) + full_img * full_mask)

		print("Postprocessing complete")

		return img, orig_faces, enhanced_faces

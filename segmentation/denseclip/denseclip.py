# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging # Use standard logging
import numpy as np
from collections import OrderedDict # Added for state dict filtering

# Explicitly import necessary components from within the denseclip package
# Assuming these are defined in a sub-module named 'models' or similar
from .models import (
    CLIPResNet, CLIPTextEncoder, CLIPVisionTransformer,
    CLIPResNetWithAttention, CLIPTextContextEncoder, ContextDecoder, ViTFeatureFusionNeck, ConvBNReLU
)


# Setup logger for this module
logger = logging.getLogger(__name__)


try:
    from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
    from torchvision.models.segmentation.fcn import FCNHead # Example replacement for FPNHead
    TORCHVISION_AVAILABLE = True
    logger.info("Successfully imported FeaturePyramidNetwork and FCNHead from torchvision.")
except ImportError:
    TORCHVISION_AVAILABLE = False
    FeaturePyramidNetwork = None
    FCNHead = None
    # Define dummy placeholders if torchvision is unavailable, to avoid errors if configs mention them
    class FeaturePyramidNetwork(nn.Module): 
         def __init__(self, **kwargs): 
               super().__init__(); self.dummy = nn.Identity()
    class FCNHead(nn.Module): 
         def __init__(self, **kwargs): 
          super().__init__(); self.dummy = nn.Identity()
    class LastLevelMaxPool(nn.Module): 
         def forward(self, *args): 
              return [torch.zeros(1)] # Needs a dummy forward
    class ViTFeatureFusionNeck(nn.Module): pass
    logger.warning("Warning: torchvision not found or FPN/FCNHead not available. Neck and Decode Head using dummy placeholders.")


# Import tokenize from the utils module
from .utils import tokenize

# Configure logging (can be done once at the top level of your application)
# logging.basicConfig(level=logging.INFO) # Example basic config

# ================== REPLACEMENTS FOR MMSEG/MMCV (Keep as is) ================== #
# class Registry ...
# SEGMENTORS = Registry()
# def resize(...)
# def add_prefix(...)
# class BaseSegmentor(...)
# ================== END REPLACEMENTS ================== #


#@SEGMENTORS.register_module() # Decorator might not be needed if SEGMENTORS registry isn't used
class DenseCLIP(nn.Module): # Inherit directly from nn.Module
    """
    DenseCLIP segmentor implementation without mmsegmentation dependencies.
    Includes CLIP pre-trained weight loading.
    """
    def __init__(self,
                 backbone, # Config dict for backbone
                 text_encoder, # Config dict for text encoder
                 decode_head, # Config dict for decode head
                 class_names, # List of class names
                 context_length,
                 # --- Arguments with Defaults ---
                 context_decoder=None, # Optional config dict
                 neck=None, # <<< Neck config dict (e.g., ViTFeatureFusionNeck)
                 context_feature='attention',
                 score_concat_index=3,
                 text_head=False, # Whether to feed text embeddings to decode head
                 tau=0.07,
                 auxiliary_head=None, # Optional config dict
                 identity_head=None, # Optional config dict
                 train_cfg=None, # Keep for potential future use
                 test_cfg=None, # Keep for potential future use
                 token_embed_dim=512, # Usually related to text token embedding before transformer
                 text_dim=512, # <<< Target dimension (e.g., 512 to match CLIP text)
                 clip_pretrained_path=None, # <<< Path to CLIP weights <<<
                 **kwargs): # Use kwargs for flexibility
        super().__init__() # Call nn.Module's init

        # --- Store basic attributes ---
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.context_length = context_length
        self.context_feature = context_feature
        self.score_concat_index = score_concat_index
        self.text_head = text_head
        self.tau = tau
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.align_corners = False # Default, updated by decode_head config


        # --- Build Backbone ---
        # Create a copy to avoid modifying the original config dict passed in
        backbone_cfg_copy = backbone.copy() # Use copy for modifications
        backbone_type = backbone_cfg_copy.pop('type')
        logger.info(f"Building backbone: {backbone_type} with config: {backbone_cfg_copy}")
        if backbone_type == "CLIPResNet":
             self.backbone = CLIPResNet(**backbone_cfg_copy)
             backbone_out_channels = backbone.get('width', 64) * 8 * 4 # Use original dict
        elif backbone_type == "CLIPResNetWithAttention":
             self.backbone = CLIPResNetWithAttention(**backbone_cfg_copy)
             backbone_out_channels = backbone.get('output_dim', 1024) # Use original dict
        elif backbone_type == "CLIPVisionTransformer":
              self.backbone = CLIPVisionTransformer(**backbone_cfg_copy)
              backbone_out_channels = backbone.get('width', 768) # Use original dict
              # Check consistency, using original dict again
              if backbone_out_channels != backbone.get('output_dim', backbone_out_channels): logger.warning(...)
        # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
        else: raise ValueError(f"Unsupported backbone type: {backbone_type}")
        logger.info(f"Built backbone. Output channels/dim: {backbone_out_channels}")


        # --- Build Text Encoder ---
        text_encoder_cfg_copy = text_encoder.copy() # Use copy
        text_encoder_type = text_encoder_cfg_copy.pop('type')
        logger.info(f"Building text encoder: {text_encoder_type} with config: {text_encoder_cfg_copy}")
        text_encoder_out_dim = text_encoder.get('embed_dim', text_dim) # Use original dict
        # Ensure internal variable 'text_dim' matches the encoder's configured output
        if text_encoder_out_dim != text_dim:
             logger.warning(f"Model 'text_dim' ({text_dim}) != text_encoder 'embed_dim' ({text_encoder_out_dim}). Using text_encoder 'embed_dim'.")
             text_dim = text_encoder_out_dim # Align internal text_dim with encoder output
        if text_encoder_type == "CLIPTextEncoder":
             self.text_encoder = CLIPTextEncoder(**text_encoder_cfg_copy)
        elif text_encoder_type == "CLIPTextContextEncoder":
             self.text_encoder = CLIPTextContextEncoder(**text_encoder_cfg_copy)
        else: raise ValueError(f"Unsupported text_encoder type: {text_encoder_type}")
        logger.info(f"Built text encoder. Output dim: {text_dim}")


        # --- Load Pre-trained CLIP Weights ---
        if clip_pretrained_path:
            logger.info(f"Attempting to load pre-trained CLIP weights from: {clip_pretrained_path}")
            try:
                logger.info("Loading TorchScript model...")
                clip_model_jit = torch.jit.load(clip_pretrained_path, map_location="cpu")
                logger.info("TorchScript model loaded successfully.")
                clip_state_dict = clip_model_jit.state_dict()
                logger.info(f"Extracted state_dict with {len(clip_state_dict)} keys.")

                # Prepare and load Visual Weights
                visual_weights = OrderedDict(); visual_prefix = 'visual.'; count_visual = 0
                for k, v in clip_state_dict.items():
                    if k.startswith(visual_prefix): visual_weights[k[len(visual_prefix):]] = v; count_visual += 1
                if visual_weights:
                    logger.info(f"Loading {count_visual} keys into visual backbone (strict=False)...")
                    load_msg_visual = self.backbone.load_state_dict(visual_weights, strict=False)
                    logger.info(f"Visual backbone loading message: {load_msg_visual}")
                    if load_msg_visual.missing_keys: logger.warning(f"Visual backbone MISSING keys: {load_msg_visual.missing_keys}")
                    if load_msg_visual.unexpected_keys: logger.warning(f"Visual backbone UNEXPECTED keys: {load_msg_visual.unexpected_keys}")
                else: logger.warning(f"No keys matching '{visual_prefix}' prefix found...")

                # Prepare and load Text Weights
                text_weights = OrderedDict(); text_prefixes_or_keys = ('transformer.', 'token_embedding.', 'positional_embedding', 'ln_final.', 'text_projection'); count_text = 0
                for k, v in clip_state_dict.items():
                    if k.startswith(text_prefixes_or_keys): text_weights[k] = v; count_text += 1
                if text_weights:
                    logger.info(f"Loading {count_text} keys into text encoder (strict=False)...")
                    load_msg_text = self.text_encoder.load_state_dict(text_weights, strict=False)
                    logger.info(f"Text encoder loading message: {load_msg_text}")
                    if load_msg_text.missing_keys: logger.warning(f"Text encoder MISSING keys: {load_msg_text.missing_keys}")
                    if load_msg_text.unexpected_keys: logger.warning(f"Text encoder UNEXPECTED keys: {load_msg_text.unexpected_keys}")
                else: logger.warning("No keys matching typical text encoder prefixes/names found...")

                del clip_model_jit, clip_state_dict # Free memory
                logger.info("CLIP jit model and state_dict deleted from memory after loading.")

            except FileNotFoundError: logger.error(f"Pre-trained CLIP file not found: {clip_pretrained_path}")
            except Exception as e: logger.error(f"Error loading pre-trained CLIP weights: {e}", exc_info=True)
        else:
            logger.warning("No 'clip_pretrained_path' provided, weights not loaded from CLIP file.")


        # --- Add Visual Projection Layers IF needed ---
        self.vis_proj = None
        self.global_proj = None
        if backbone_out_channels != text_dim:
            logger.info(f"Backbone out dim ({backbone_out_channels}) != Text dim ({text_dim}). Adding projection layers.")
            self.vis_proj = nn.Conv2d(backbone_out_channels, text_dim, kernel_size=1)
            self.global_proj = nn.Linear(backbone_out_channels, text_dim)
        else: logger.info(f"Backbone/Text dims match ({text_dim}). No projection needed.")


        # --- Build Context Decoder ---
        self.context_decoder = None
        if context_decoder:
            context_decoder_cfg = context_decoder.copy(); cd_type = context_decoder_cfg.pop('type'); logger.info(f"Building ContextDecoder: {cd_type}...")
            if cd_type == "ContextDecoder": cd_cfg.setdefault('visual_dim', text_dim); self.context_decoder = ContextDecoder(**cd_cfg)
            else: raise ValueError(...)
        else: logger.info("No context decoder configured.")


        # --- Build Neck ---
        self.neck = None
        self._neck_out_keys = None
        # Determine the input dimension for the next stage (head or proj layer if no neck)
        head_in_channels = backbone_out_channels # Start with raw backbone output dim
        if self.vis_proj is not None: head_in_channels = text_dim # If no neck, head gets projected features? NO - head usually gets BB features

        if neck: # Check if neck config is provided
             neck_cfg = neck.copy()
             neck_type = neck_cfg.pop('type')
             logger.info(f"Building neck: {neck_type} with config: {neck_cfg}")
             # --- VVVVV Instantiate Neck based on Type VVVVV ---
             if neck_type == "ViTFeatureFusionNeck":
                  in_channels_list = neck.get('in_channels_list')
                  out_channels = neck.get('out_channels')
                  if in_channels_list is None or out_channels is None: raise ValueError("Neck config requires 'in_channels_list' and 'out_channels'")
                  # Import the class (ensure it's importable from models.py)
                  from .models import ViTFeatureFusionNeck
                  self.neck = ViTFeatureFusionNeck(
                      in_channels_list=in_channels_list,
                      out_channels=out_channels,
                      inter_channels=neck_cfg.get('inter_channels') # Pass optional arg
                  )
                  head_in_channels = out_channels # <<< UPDATE head input dim based on neck output <<<
                  logger.info(f"Built ViTFeatureFusionNeck. Output channels for head: {head_in_channels}")

             elif neck_type == "FPN" and FeaturePyramidNetwork is not None:
                  logger.info("Building standard FPN neck...")
                  # Default FPN in_channels based on ResNet convention
                  default_fpn_in = [ backbone.get('width', 64) * 2**(i+1) * 4 for i in range(4)] # Use original 'backbone' dict
                  in_channels_list = neck.get('in_channels', default_fpn_in) # Use original 'neck' dict
                  out_channels = neck.get('out_channels', 256) # Use original 'neck' dict
                  num_outs = neck.get('num_outs', len(in_channels_list)) # Use original 'neck' dict
                  extra_blocks = None
                  if num_outs > len(in_channels_list):
                       if LastLevelMaxPool: extra_blocks = LastLevelMaxPool(); logger.info("Adding LastLevelMaxPool...")
                       else: logger.warning("LastLevelMaxPool not available...")
                  # Add checks for list and positive int
                  if not isinstance(in_channels_list, list) or not in_channels_list: raise ValueError(...)
                  if not isinstance(out_channels, int) or out_channels <= 0: raise ValueError(...)
                  self.neck = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels, extra_blocks=extra_blocks)
                  head_in_channels = out_channels # <<< UPDATE head input dim based on neck output <<<
                  self._neck_out_keys = [str(i) for i in range(num_outs)] # Assuming default keys '0', '1', ...
                  logger.info(f"Built torchvision FPN. Output channels for head: {head_in_channels}. Assumed keys: {self._neck_out_keys}")
             elif neck_type == "FPN": logger.error("Torchvision FPN specified but not available.")
             else: raise ValueError(f"Unsupported neck type: {neck_type}")
             # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
        else:
             # If no neck, determine head input channels carefully
             # Head usually takes last stage output of backbone
             head_in_channels = backbone_out_channels
             logger.info(f"No neck configured. Head will receive features directly from backbone with {head_in_channels} channels.")


        # --- Build Decode Head ---
        self.decode_head = None
        self._decode_head_cfg = None
        if decode_head:
            decode_head_cfg_copy = decode_head.copy() # Use copy for instantiation
            decode_head_type = decode_head_cfg_copy.pop('type')
            logger.info(f"Building decode head: {decode_head_type}...")
            # --- VVVVV Use original 'decode_head' dict for .get() VVVVV ---
            self.align_corners = decode_head.get('align_corners', False)
            self.num_classes = decode_head.get('num_classes', self.num_classes)
            in_channels_cfg = decode_head.get('in_channels')
            # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
            final_head_in_channels = head_in_channels # Start with inferred channels
            if in_channels_cfg is not None:
                if in_channels_cfg != head_in_channels: logger.warning(...); final_head_in_channels = in_channels_cfg
            logger.info(f"Decode head using final input channels: {final_head_in_channels}")

            if decode_head_type == "FPNHead" and FCNHead is not None:
                 # --- VVVVV Use original 'decode_head' dict for .get() VVVVV ---
                 channels = decode_head.get('channels', 256)
                 # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
                 self.decode_head = FCNHead(in_channels=final_head_in_channels, channels=channels)
                 self.decode_head.classifier = nn.Conv2d(channels, self.num_classes, kernel_size=1)
                 logger.info(f"Built torchvision FCNHead...")
            # --- VVVVV Use original 'decode_head' dict for .get() VVVVV ---
            elif decode_head_type == "ViTSegmentationDecoder":
                 from .heads import ViTSegmentationDecoder
                 encoder_channels = decode_head.get('encoder_channels') # Use original dict
                 decoder_channels = decode_head.get('decoder_channels') # Use original dict
            # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---
                 if encoder_channels is None or decoder_channels is None: raise ValueError(...)
                 self.decode_head = ViTSegmentationDecoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=self.num_classes, align_corners=self.align_corners)
                 logger.info("Built ViTSegmentationDecoder.")
            elif decode_head_type == "IdentityHead" and IDENTITY_HEAD_AVAILABLE: self.decode_head = IdentityHead(**decode_head_cfg); logger.info("Built IdentityHead.")
            elif decode_head_type == "IdentityHead": raise ValueError("IdentityHead class definition not found.")
            else: raise ValueError(f"Unsupported decode_head type: {decode_head_type}")

        self.with_decode_head = self.decode_head is not None
        if not self.with_decode_head: logger.warning("No decode head was built.")


        # --- Build Auxiliary Head (Keep Removed/Commented Out for ViT) ---
        self.auxiliary_head = None; self.with_auxiliary_head = False
        if auxiliary_head: logger.warning("Auxiliary head configured but not built for ViT setup.")


        # --- Build Identity Head (Keep logic) ---
        self.identity_head = None; self.with_identity_head = False
        if identity_head: # ... (logic as before) ...
            pass


        # --- Tokenization and Learnable Parameters ---
        logger.info(f"Tokenizing {len(self.class_names)} class names...")
        try:
             self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in self.class_names])
        except NameError: logger.error("'tokenize' function not imported/defined!"); raise
        # ... (prompt length calculation) ...
        if not hasattr(self, 'text_encoder'): raise RuntimeError("Text encoder not initialized before prompt calculation.")
        text_encoder_context_length = getattr(self.text_encoder, 'context_length', 77)
        logger.info(f"Text encoder context length capacity: {text_encoder_context_length}")
        if self.context_length > text_encoder_context_length: logger.warning(...); self.context_length = text_encoder_context_length
        prompt_context_length = text_encoder_context_length - self.context_length
        # ... (context/gamma initialization) ...
        if isinstance(self.text_encoder, CLIPTextContextEncoder):
             _token_embed_dim = text_encoder.get('transformer_width', token_embed_dim) # Use original dict
             _text_dim_gamma = text_dim
             if prompt_context_length > 0: self.contexts = nn.Parameter(...); nn.init.trunc_normal_(...); logger.info(...)
             else: self.contexts = None; logger.info("No space for learnable contexts.")
             self.gamma = nn.Parameter(torch.ones(_text_dim_gamma) * 1e-4); logger.info(...)
        else: self.contexts = None; self.gamma = None; logger.info("Standard text encoder used...")


        # --- Custom Weight Initialization for Non-CLIP parts ---
        logger.info("Applying custom weight initialization to non-CLIP modules...")
        self._init_non_clip_weights() # Call the helper function


    def _init_weights_fn(self, m):
        """Helper function for applying initialization."""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
             # Kaiming initialization is common for Conv layers in segmentation heads/necks
             try: # Use try-except for robustness
                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                 if m.bias is not None:
                     nn.init.constant_(m.bias, 0)
                 # logger.debug(f"Initialized {classname} with Kaiming normal.")
             except AttributeError: pass # Skip modules without weight/bias like pooling
        elif classname.find('Linear') != -1:
             try:
                 nn.init.normal_(m.weight, 0, 0.01)
                 if m.bias is not None:
                     nn.init.constant_(m.bias, 0)
                 # logger.debug(f"Initialized {classname} with Normal(0, 0.01).")
             except AttributeError: pass
        elif classname.find('BatchNorm') != -1: # Covers BatchNorm1d, BatchNorm2d etc.
             try:
                 nn.init.constant_(m.weight, 1)
                 nn.init.constant_(m.bias, 0)
                 # logger.debug(f"Initialized {classname} with constants (1, 0).")
             except AttributeError: pass
        elif classname.find('GroupNorm') != -1:
             try:
                 nn.init.constant_(m.weight, 1)
                 nn.init.constant_(m.bias, 0)
                 # logger.debug(f"Initialized {classname} with constants (1, 0).")
             except AttributeError: pass
        # Add other layer types if needed (e.g., LayerNorm)

    def _init_non_clip_weights(self):
        """Initialize weights for modules NOT loaded from CLIP."""
        logger.info("Applying custom initialization using _init_weights_fn...")
        modules_to_init = []
        # Add newly added projection layers
        if self.vis_proj is not None: modules_to_init.append(('vis_proj', self.vis_proj))
        if self.global_proj is not None: modules_to_init.append(('global_proj', self.global_proj)) # Include global_proj
        # Add other trainable components
        if self.context_decoder is not None: modules_to_init.append(('context_decoder', self.context_decoder))
        # Initialize neck ONLY if it exists (should be None for ViT default)
        if self.neck is not None: modules_to_init.append(('neck', self.neck))
        if self.decode_head is not None: modules_to_init.append(('decode_head', self.decode_head))
        # Initialize aux head ONLY if it exists (should be None for ViT default)
        if self.auxiliary_head is not None: modules_to_init.append(('auxiliary_head', self.auxiliary_head))
        # Handle identity_head only if it's a separate module instance
        if self.with_identity_head and self.identity_head is not None and self.identity_head is not self.decode_head:
            modules_to_init.append(('identity_head', self.identity_head))

        for name, module in modules_to_init:
             # Prevent applying to CLIP backbone/text encoder if somehow listed
             if name in ['backbone', 'text_encoder']: continue
             logger.info(f"Initializing module: {name}...")
             module.apply(self._init_weights_fn)
             # Special handling for final classifier layers in heads
             if name in ['decode_head', 'auxiliary_head'] and hasattr(module, 'classifier') and isinstance(module.classifier, nn.Conv2d):
                 logger.info(f"...Initializing final classifier of {name} with Normal(0, 0.01)...")
                 nn.init.normal_(module.classifier.weight, mean=0, std=0.01)
                 if module.classifier.bias is not None:
                      nn.init.constant_(module.classifier.bias, 0)

    # --- MODIFIED extract_feat ---
    def extract_feat(self, img):
        """
        Extract features from images using the backbone.
        Assumes backbone returns a list of tensors.
        """
        logger.debug("Extracting features with backbone...")
        features = self.backbone(img) # Call the backbone's forward method

        # --- Detailed Debug Print ---
        logger.debug(f"Raw backbone output type: {type(features)}")
        if isinstance(features, (list, tuple)):
             logger.debug(f"Raw backbone output length: {len(features)}")
             if features: # Check if not empty
                # Log shape of first element for verification
                logger.debug(f"  Element 0 type: {type(features[0])}, shape: {features[0].shape if isinstance(features[0], torch.Tensor) else 'N/A'}")
        elif isinstance(features, torch.Tensor):
             logger.debug(f"Raw backbone output shape: {features.shape}")
        # --- End Detailed Debug Print ---


        # --- VVVVV FINAL SIMPLIFIED LOGIC VVVVV ---
        # Check if the output is a list or tuple
        if isinstance(features, (list, tuple)):
            # Check if it's NOT empty
            if not features:
                logger.error("Backbone returned an empty list/tuple.")
                return []

            # Check if ALL elements are Tensors (more robust check)
            if not all(isinstance(f, torch.Tensor) for f in features):
                 logger.error(f"Backbone output list/tuple contains non-Tensor elements: {[type(f) for f in features]}")
                 return []

            # Optional: Check if all elements are 4D (expected for spatial features)
            if not all(f.ndim == 4 for f in features):
                 logger.warning(f"Backbone output list contains non-4D Tensors: {[f.ndim for f in features]}. Check backbone output.")
                 # Depending on neck/head, might need error here, but let's allow for now

            # If checks pass, return the full list of features
            logger.debug(f"Backbone returned a list/tuple of {len(features)} feature tensors.")
            return list(features) # Return the validated list

        elif isinstance(features, torch.Tensor) and features.ndim == 4:
             # Handle case where backbone *directly* returned a 4D tensor
             logger.warning("Backbone returned single tensor directly instead of list/tuple. Wrapping in list.")
             return [features]
        else:
             # Handle completely unexpected output formats
             logger.error(f"Unknown or unsupported backbone output format: {type(features)}. Returning empty list.")
             return []
        # --- ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ---

        
    # --- _process_features ---
    def _process_features(self, x):
        """
        Handles feature processing after backbone extraction.
        Applies projection, calculates text features, context fusion, and score map.
        Adapts for single feature map input from ViT (after extract_feat).
        Returns:
            text_embeddings (Tensor): Shape [B, K, C_text]
            features_for_head (list[Tensor]): Original backbone feature maps (before score map concat).
                                             For ViT, this will be a list with one tensor.
            score_map (Tensor): Shape [B, K, H_vis, W_vis] (using potentially projected visual features).
            _x_orig (list[Tensor]): Copy of original backbone feature maps list.
        """
        # --- Input Validation ---
        if not isinstance(x, (list, tuple)) or not x:
            raise ValueError(f"Expected _process_features input 'x' to be a non-empty list/tuple. Got: {type(x)}")

        _x_orig = [feat.clone() for feat in x] # Keep original features list (clone for safety)

        # --- Extract Global and Spatial Features ---
        # For ViT, x is likely [_ViT_spatial_map_], so x[-1] is the main spatial feature
        # For ResNet, x is [stage1, ..., stage4], x[-1] is last stage spatial feature
        visual_embeddings = x[-1] # Assume last element is the primary spatial map
        if visual_embeddings.ndim != 4:
            raise ValueError(f"Expected last backbone feature map to be 4D, got {visual_embeddings.ndim}D")

        # Calculate global feature by pooling the last spatial map
        global_feat = F.adaptive_avg_pool2d(visual_embeddings, (1, 1)).flatten(1)
        logger.debug(f"Calculated global_feat shape: {global_feat.shape}")

        B, C_vis_orig, H_vis, W_vis = visual_embeddings.shape
        C_glob_orig = global_feat.shape[1]
        if C_vis_orig != C_glob_orig:
            logger.warning(f"Initial spatial feature channels ({C_vis_orig}) != global feature channels ({C_glob_orig}). Check backbone output or pooling.")

        # --- Apply Global Projection (If layer exists) ---
        if self.global_proj is not None:
             logger.debug(f"Applying global projection: {C_glob_orig} -> {self.global_proj.out_features}")
             global_feat = self.global_proj(global_feat) # Project [B, C_orig] -> [B, C_proj]
             C_glob = global_feat.shape[1]
             logger.debug(f"Projected global_feat shape: {global_feat.shape}")
        else:
             C_glob = C_glob_orig

        # --- Apply Visual Spatial Projection (If layer exists) ---
        if self.vis_proj is not None:
             logger.debug(f"Applying visual spatial projection: {C_vis_orig} -> {self.vis_proj.out_channels}")
             visual_embeddings = self.vis_proj(visual_embeddings) # Project [B, C_vis_orig, H, W] -> [B, C_vis, H, W]
             B, C_vis, H_vis, W_vis = visual_embeddings.shape
             logger.debug(f"Projected spatial visual_embeddings shape: {visual_embeddings.shape}")
        else:
             C_vis = C_vis_orig

        # --- Check consistency AFTER projections ---
        if C_vis != C_glob:
             logger.error(f"Projected spatial dim C_vis ({C_vis}) != projected global dim C_glob ({C_glob}). Check projection layers.")

        # --- Prepare Visual Context for Context Decoder ---
        visual_context = None
        if self.context_decoder:
            if self.context_feature == 'attention':
                # Use projected global feature + projected spatial features
                global_feat_ctx = global_feat # Should have correct projected dimension now
                visual_context = torch.cat([global_feat_ctx.unsqueeze(1), visual_embeddings.flatten(2).permute(0, 2, 1)], dim=1)
                logger.debug(f"Prepared visual context ('attention') shape: {visual_context.shape}")

            elif self.context_feature == 'backbone':
                 # Use the *potentially projected* spatial features for consistency if vis_proj exists
                 visual_context_spatial = visual_embeddings # Use features AFTER vis_proj
                 C_context = visual_context_spatial.shape[1]
                 visual_context = visual_context_spatial.flatten(2).permute(0, 2, 1) # [B, H*W, C_vis]
                 # Check dimension against decoder's expected dim
                 if hasattr(self.context_decoder, 'visual_dim') and C_context != self.context_decoder.visual_dim:
                     logger.warning(f"Context feature 'backbone': Feature dim ({C_context}) != context_decoder expected dim ({self.context_decoder.visual_dim}).")
                 logger.debug(f"Prepared visual context ('backbone' using spatial feats) shape: {visual_context.shape}")

            else: raise ValueError(f"Invalid context_feature type: {self.context_feature}")


        # --- Text Feature Calculation ---
        if not hasattr(self, 'text_encoder'): raise AttributeError("text_encoder missing")
        text_embeddings_device = next(self.text_encoder.parameters()).device
        tokenized_texts = self.texts.to(text_embeddings_device)
        if isinstance(self.text_encoder, CLIPTextContextEncoder) and self.contexts is not None:
             contexts_device = self.contexts.to(text_embeddings_device)
             text_embeddings = self.text_encoder(tokenized_texts, contexts_device).expand(B, -1, -1)
        elif isinstance(self.text_encoder, CLIPTextEncoder):
             text_embeddings = self.text_encoder(tokenized_texts).expand(B, -1, -1)
        else: raise TypeError(...)
        logger.debug(f"Raw text embeddings shape: {text_embeddings.shape}")

        # Apply Context Decoder Fusion
        if self.context_decoder and visual_context is not None:
            if self.gamma is None: raise AttributeError(...)
            logger.debug(f"Applying context decoder...")
            visual_context_device = visual_context.to(text_embeddings_device)
            try: text_diff = self.context_decoder(text_embeddings, visual_context_device); gamma_device = self.gamma.to(text_embeddings_device); text_embeddings = text_embeddings + gamma_device * text_diff; logger.debug("Applied context fusion.")
            except Exception as cd_e: logger.error(...) ; raise cd_e
        elif self.context_decoder and visual_context is None: logger.error("Context decoder configured but no visual context.")


        # --- Score Map Calculation ---
        B, K, C_text = text_embeddings.shape
        visual_norm = F.normalize(visual_embeddings, dim=1, p=2) # Use potentially projected spatial features
        text_norm = F.normalize(text_embeddings, dim=2, p=2)
        if C_vis != C_text: raise ValueError(f"Visual dim after proj ({C_vis}) != Text dim ({C_text}).")
        score_map = torch.einsum('bchw,bkc->bkhw', visual_norm, text_norm)
        logger.debug(f"Calculated score map shape: {score_map.shape}")


        # --- Feature Concatenation ---
        # features_for_head will be used if neck is None, otherwise _x_orig goes to neck
        # Let's return _x_orig as features_for_head consistently and handle concat later if needed
        features_for_head = _x_orig # Return original backbone features list

        if 0 <= self.score_concat_index < len(features_for_head):
            # Apply concat to the COPY we are returning
            target_feat_map = features_for_head[self.score_concat_index]
            logger.warning(f"Applying score map concatenation at index {self.score_concat_index}. Ensure neck/head handles this.")
            try:
                score_map_resized = F.interpolate(score_map, size=target_feat_map.shape[2:], mode='bilinear', align_corners=False)
                features_for_head[self.score_concat_index] = torch.cat([target_feat_map, score_map_resized], dim=1)
                logger.info(f"Concatenated score map. New shape at index {self.score_concat_index}: {features_for_head[self.score_concat_index].shape}")
            except Exception as concat_e: logger.error(...) ; logger.warning("Proceeding without concat due to error.")
        elif self.score_concat_index != -1: # Only warn if index is not explicitly disabled (-1)
            logger.warning(f"score_concat_index {self.score_concat_index} invalid. Score map not concatenated.")


        # Return: text embeddings, features for neck/head, score map, original backbone features
        return text_embeddings, features_for_head, score_map, _x_orig


    # --- forward ---
    def forward(self, img, img_metas=None, gt_semantic_seg=None, return_loss=True, **kwargs):
        """
        Main forward pass. Handles backbone, neck (optional), head,
        and text/visual feature processing. Correctly selects input for decode head.
        """
        # 1. Extract Features from Backbone
        # Returns a list of tensors, e.g., [stage1, ..., stage4] for ResNet
        # or [spatial_map_768] for modified ViT
        backbone_features = self.extract_feat(img)
        if not backbone_features: # Handle failure
            logger.error("Backbone feature extraction failed or returned empty.")
            if return_loss and self.training: return {'main_output': None, 'aux_losses': {}}
            else: return None

        # Keep a copy for potential auxiliary head use
        # Note: Using _x_orig assumes extract_feat returns features BEFORE any projection
        # that might happen in _process_features. This is usually correct.
        _x_orig = [feat.clone() for feat in backbone_features]

        # 2. Process Features (for Score Map & Context Decoder)
        # This primarily calculates text_embeddings and score_map.
        # It uses the *last* original backbone feature (_x_orig[-1]) internally for processing.
        # We don't need the 'features_for_head' return value from it here anymore.
        text_embeddings, _, score_map, _ = self._process_features(_x_orig)


        # 3. Process Features through Neck (if exists)
        # Select the correct input for the head based on whether a neck exists
        if self.neck:
            # --- Neck Path ---
            logger.debug("Passing ORIGINAL backbone features (_x_orig) through neck...")
            # Neck expects original multi-scale features
            # Create dict only if needed by neck (e.g., torchvision FPN)
            if isinstance(self.neck, FeaturePyramidNetwork):
                 neck_input = {str(i): feat for i, feat in enumerate(_x_orig)}
            else: # Assume neck takes list directly
                 neck_input = _x_orig

            features_after_neck = self.neck(neck_input) # Neck should return list: [fused_map] or [p2,p3,p4,p5]

            if not features_after_neck:
                logger.error("Neck processing returned empty features.")
                if return_loss and self.training: return {'main_output': None, 'aux_losses': {}}
                else: return None
            logger.debug(f"Neck output shapes: {[f.shape for f in features_after_neck]}")
            # Determine the input tensor for the head from the neck output
            if isinstance(features_after_neck, (list, tuple)) and len(features_after_neck) > 0:
                 # If FPN neck was used (output keys '0','1','2',...), use highest res (index 0)
                 # If ViTFeatureFusionNeck was used (output list [fused_map]), use index 0
                 input_for_decode_head = features_after_neck[0] # <<< Use FIRST feature from neck output
                 logger.debug(f"Using neck output feature 0 (shape {input_for_decode_head.shape}) for decode head.")
            else:
                 logger.error(f"Could not determine valid feature tensor from neck output: {type(features_after_neck)}")
                 input_for_decode_head = None
            # --- End Neck Path ---
        else:
            # --- No Neck Path ---
            # Use the LAST feature map directly from the backbone output (_x_orig)
            # This tensor has the backbone's output dimension (e.g., 768 for ViT)
            if not _x_orig: # Should have been caught earlier, but check again
                logger.error("Original backbone features (_x_orig) are empty.")
                input_for_decode_head = None
            else:
                input_for_decode_head = _x_orig[-1] # <<< Use LAST feature from backbone
                logger.debug(f"Skipping neck. Using last backbone feature (shape {input_for_decode_head.shape}) for decode head.")
            # --- End No Neck Path ---


        # 4. Prepare Input for Auxiliary Head (if exists)
        input_for_aux_head = None
        if self.with_auxiliary_head:
             # Needs logic based on which feature from _x_orig the aux head expects
             aux_input_index = self._decode_head_cfg.get('aux_input_index', 2) # Example default
             if 0 <= aux_input_index < len(_x_orig):
                  input_for_aux_head = _x_orig[aux_input_index]
                  logger.debug(f"Using original feature idx {aux_input_index} for aux head. Shape: {input_for_aux_head.shape}")
             else:
                  logger.warning(f"Cannot get feature at index {aux_input_index} for auxiliary head with current backbone output (length {len(_x_orig)}).")


        # 5. Forward through Decode Head(s)
        if not self.decode_head:
            raise RuntimeError("Decode head is not defined.")
        if input_for_decode_head is None: # Handle error case from step 3/4
             logger.error("Input tensor for decode head is None. Cannot proceed.")
             if return_loss and self.training: return {'main_output': None, 'aux_losses': {}}
             else: return None

        # Pass the selected single tensor to the main decode head
        output_logits = self.decode_head(input_for_decode_head)
        logger.debug(f"Main decode head output shape: {output_logits.shape if output_logits is not None else 'None'}")


        # 6. Handle Training vs Inference
        if return_loss and self.training:
             if gt_semantic_seg is None:
                  raise ValueError("gt_semantic_seg is required for training (return_loss=True)")

             losses = {} # Dictionary to store auxiliary losses/logits

             # --- Main Output Logits ---
             gt_h, gt_w = gt_semantic_seg.shape[-2:]
             output_logits_resized = output_logits # Default to non-resized
             if output_logits is not None and output_logits.shape[-2:] != (gt_h, gt_w):
                  output_logits_resized = F.interpolate(
                      output_logits, size=(gt_h, gt_w), mode='bilinear', align_corners=self.align_corners
                  )
                  logger.debug(f"Resized main logits to GT shape: {output_logits_resized.shape}")

             # --- Auxiliary Head Logits ---
             if self.with_auxiliary_head and self.auxiliary_head and input_for_aux_head is not None:
                  try:
                      aux_logits = self.auxiliary_head(input_for_aux_head)
                      aux_logits_resized = aux_logits
                      if aux_logits.shape[-2:] != (gt_h, gt_w):
                           aux_logits_resized = F.interpolate(aux_logits, size=(gt_h, gt_w), mode='bilinear', align_corners=self.align_corners)
                      losses['aux_output'] = aux_logits_resized
                      logger.debug(f"Aux head output logits (resized) shape: {aux_logits_resized.shape}")
                  except Exception as aux_e:
                      logger.error(f"Error during auxiliary head forward: {aux_e}", exc_info=True)

             # --- Identity Head Logits ---
             if self.with_identity_head and self.identity_head:
                  try:
                      id_input = score_map / self.tau # Use score_map calculated earlier
                      id_logits = self.identity_head(id_input)
                      id_logits_resized = id_logits
                      if id_logits.shape[-2:] != (gt_h, gt_w):
                           id_logits_resized = F.interpolate(id_logits, size=(gt_h, gt_w), mode='bilinear', align_corners=self.align_corners)
                      losses['identity_output'] = id_logits_resized
                      logger.debug(f"Identity head output logits (resized) shape: {id_logits_resized.shape}")
                  except Exception as id_e:
                      logger.error(f"Error during identity head forward: {id_e}", exc_info=True)

             # Return main logits (potentially resized) and dict of auxiliary logits
             return {'main_output': output_logits_resized, 'aux_losses': losses}

        else: # Inference mode (return_loss=False)
             if output_logits is None: return None

             # Resize final output logits to match the original input image size
             logger.debug(f"Resizing inference output {output_logits.shape} to image shape {img.shape[2:]}")
             output = F.interpolate(
                 input=output_logits,
                 size=img.shape[2:],
                 mode='bilinear',
                 align_corners=self.align_corners
             )
             logger.debug(f"Final inference output shape: {output.shape}")
             return output
        


    
    def _get_final_visual_embeddings(self, x):
        """ Helper to get the spatial visual features AFTER potential projection """
        if not isinstance(x, (list, tuple)) or not x: return None # Handle bad input
        visual_embeddings = x[-1] # Assume last element is spatial features
        # Handle edge case where backbone returns (global, spatial) tuple as last element
        if isinstance(visual_embeddings, (list, tuple)) and len(visual_embeddings)==2:
            visual_embeddings = visual_embeddings[1] # Take the spatial part
        if not isinstance(visual_embeddings, torch.Tensor) or visual_embeddings.ndim != 4:
            logger.error(f"Could not extract valid 4D spatial tensor in _get_final_visual_embeddings. Got: {type(visual_embeddings)}")
            return None
        # Apply projection if it exists
        if self.vis_proj is not None:
            visual_embeddings = self.vis_proj(visual_embeddings)
        return visual_embeddings


    # --- Inference Helper Methods  ---
    def inference(self, img, img_meta, rescale):
         """Simple inference, returns logits potentially rescaled to original image size."""
         # test_cfg might control sliding window etc. - not implemented here
         seg_logit = self.forward(img, img_metas=img_meta, return_loss=False) # Call main forward in inference mode

         if seg_logit is None: return None # Handle forward pass failure

         # Rescaling logic to original image shape
         if rescale and img_meta is not None and len(img_meta) > 0 and 'ori_shape' in img_meta[0]:
              ori_shape = img_meta[0]['ori_shape'][:2] # H, W
              if seg_logit.shape[2:] != tuple(ori_shape):
                  logger.debug(f"Rescaling inference output from {seg_logit.shape[2:]} to original shape {ori_shape}")
                  seg_logit = F.interpolate(
                      seg_logit,
                      size=ori_shape,
                      mode='bilinear',
                      align_corners=self.align_corners # Use head's setting
                  )
         elif rescale:
              logger.warning("Rescale=True but ori_shape not found in img_meta. Returning logits at input image size.")

         return seg_logit

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image. Returns numpy prediction."""
        seg_logit = self.inference(img, img_meta, rescale)
        if seg_logit is None: return None

        seg_pred = seg_logit.argmax(dim=1) # Get class indices [N, H, W]
        seg_pred = seg_pred.cpu().numpy()
        # Assuming batch size 1 for simple_test, return the single prediction
        return seg_pred[0] if len(seg_pred) > 0 else None

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations by averaging logits. Returns numpy prediction."""
        # imgs: list of augmented images [tensor(C,H,W), ...]
        # img_metas: list of corresponding meta dicts
        seg_logits = []
        for img, meta in zip(imgs, img_metas):
             # Add batch dimension for inference call
             logit = self.inference(img.unsqueeze(0), [meta], rescale)
             if logit is not None:
                 seg_logits.append(logit)

        if not seg_logits: return None # Handle case where all inferences failed

        # Stack logits [N_aug, C, H, W] and average
        avg_seg_logit = torch.stack(seg_logits).mean(dim=0) # [C, H, W]
        seg_pred = avg_seg_logit.argmax(dim=0) # Get class indices [H, W]
        seg_pred = seg_pred.cpu().numpy()
        return seg_pred

    
    def forward_dummy(self, img):
        """ Dummy forward for FLOPs calculation or similar. Tries to simulate main path. """
        logger.warning("forward_dummy provides a simplified path and may not accurately reflect full model complexity or handle all configurations.")
        try:
            # 1. Backbone
            x = self.extract_feat(img)
            # 2. Process (simplified - just take last feature, maybe project)
            visual_embeddings = x[-1]
            if self.vis_proj: visual_embeddings = self.vis_proj(visual_embeddings)
            features_for_head = [visual_embeddings] # Simplified input for head
            # 3. Neck (simplified - pass only last feature if neck expects list)
            if self.neck:
                neck_input = {str(len(x)-1): visual_embeddings} # Create dummy dict input
                neck_output = self.neck(neck_input)
                features_for_head = list(neck_output.values()) # Use neck output
            # 4. Head
            if self.decode_head:
                head_input = features_for_head[0] # Assume head takes first feature
                if isinstance(self.decode_head, FCNHead): # FCNHead needs dict
                     head_input = {'out': features_for_head[0]} # Provide dummy key 'out' or similar expected key
                out = self.decode_head(head_input)
                # Resize to original image size
                out = F.interpolate(input=out, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
                return out
            else:
                return features_for_head # Return intermediate features if no head
        except Exception as e:
            logger.error(f"Error during forward_dummy: {e}. Returning input image shape.")
            # Return something with expected dimension characteristics if possible
            return torch.zeros(img.shape[0], self.num_classes, *img.shape[2:], device=img.device)

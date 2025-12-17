# ============================================================================
# OVERVIEW
# ============================================================================

# Sources: 
# https://modal.com/docs/examples/flan_t5_finetune#run-via-the-cli
# Model: https://huggingface.co/lucadiliello/bart-small
# Model: https://huggingface.co/docs/transformers/en/model_doc/flan-t5

# ============================================================================
# MODEL CONFIGURATIONS FOR UNSLOTH
# ============================================================================

SMALL_MODELS = {
    'flan-t5-small': {
        'name': 'flan-t5-small', 
        'model_name': 'google/flan-t5-small',
        'params': 77_000_000,
        'max_seq_length': 512,
        'description': 'FLAN-T5 Small',   
        'color': '#3b82f6',
        'link': 'https://huggingface.co/google/flan-t5-small'
    },
    'bart-small': {
        'name': 'bart-small',
        'model_name': 'lucadiliello/bart-small',
        'params': 70_500_000,
        'max_seq_length': 512,
        'description': 'Bart Small',
        'color': '#8b5cf6',
        'link': 'https://huggingface.co/lucadiliello/bart-small'
    },
    # TODO After running both models
    # 'flan-t5-large': {
    #     'name': 'flan-t5-large',
    #     'model_name': 'google/flan-t5-large',
    #     'params': 780_000_000,
    #     'max_seq_length': 512,
    #     'description': 'FLAN-T5 Large',
    #     'color': "#00ffbb",
    #     'link': 'https://huggingface.co/google/flan-t5-large'
    # },
}

REGION_NAMES_TO_COORDINATES = {
        'us-west-2': (45.9174667, -119.2684488),    # Oregon
        'us-west-1': (37.443680, -122.153664),      # California
        'us-east-1': (38.9940541, -77.4524237)      # Virginia
    }


# Regional groupings for US cloud regions
us_west_regions = [
    'us-west-1',
    'us-west-2',
    'us-west1',
    'westus',
    'westus2',
    'westus3'
]

us_east_regions = [
    'us-east-1',
    'us-east-2',
    'us-east1',
    'us-east4',
    'eastus',
    'eastus2'
]

us_central_regions = [
    'centralus',
    'uscentral1',
    'northcentralus',
    'southcentralus',
    'westcentralus'
]

AVAILABLE_REGIONS = { 
    'us-west': us_west_regions, 
    'us-east': us_east_regions, 
    'us-central': us_central_regions                   
} 

# Regions To Test Out
# DATA_CENTERS_REGION = ['us-west', 'us-east']
# TODO: Use 'us-west', 'us-east', 'us-central' next time
TEST_REGIONS = ['us-east']


# ============================================================================
# TEST ARTICLES
# ============================================================================

TEST_ARTICLES = [
    # Deep-sea discovery (expanded)
    "Scientists have discovered a new species of deep-sea fish in the Mariana Trench. "
    "The fish, which has been named Pseudoliparis swirei, was found at a depth of "
    "8,000 meters below the surface during a month-long expedition. This discovery provides "
    "new insights into how life can adapt to extreme conditions in the ocean's deepest regions. "
    "The translucent fish, measuring just 10 centimeters in length, has evolved unique "
    "adaptations including a gelatinous body structure that withstands crushing pressure and "
    "specialized proteins that prevent cellular collapse. Researchers used advanced deep-sea "
    "submersibles equipped with high-definition cameras and sampling equipment to capture "
    "specimens. The discovery challenges previous assumptions about the limits of vertebrate "
    "life in extreme environments. Scientists believe studying these adaptations could inform "
    "biotechnology applications and help us understand life's potential on other planets with "
    "extreme conditions. The research team included marine biologists from five countries and "
    "was funded by the National Science Foundation's ocean exploration program.",
    
    # Climate change and bird migration (expanded)
    "A new study published in Nature reveals that climate change is affecting bird migration "
    "patterns across North America in unprecedented ways. Researchers analyzed historical data "
    "from over 500 bird species spanning seven decades, combining citizen science observations "
    "with satellite tracking technology. The findings show that many species are arriving at "
    "their breeding grounds earlier than they did 50 years ago, with some species advancing "
    "their arrival by as much as two weeks. This shift correlates directly with rising spring "
    "temperatures and earlier snowmelt patterns. The study also reveals concerning mismatches "
    "between bird arrival times and peak insect abundance, potentially affecting breeding "
    "success. Songbirds like warblers and thrushes show the most dramatic shifts, while "
    "waterfowl display more variable patterns. Scientists warn these changes could lead to "
    "population declines if birds cannot adapt quickly enough to shifting food availability. "
    "The research emphasizes the urgent need for conservation strategies that account for "
    "climate-driven ecological disruptions and highlights the importance of protecting "
    "migratory corridors and breeding habitats.",
    
    # AI advancement (expanded)
    "The latest artificial intelligence model from OpenAI has demonstrated unprecedented "
    "capabilities in natural language understanding and generation, marking a significant "
    "milestone in AI development. The model, built on transformer architecture with 175 "
    "billion parameters, can perform complex reasoning tasks and maintain coherent conversations "
    "across multiple topics while showing improved accuracy in factual responses. During "
    "extensive testing, the AI successfully completed advanced mathematical proofs, generated "
    "creative writing indistinguishable from human authors, and provided expert-level analysis "
    "in specialized domains like medicine and law. The system employs reinforcement learning "
    "from human feedback to align its outputs with user intentions and ethical guidelines. "
    "Researchers implemented safety measures including content filtering and bias mitigation "
    "protocols to address concerns about potential misuse. The model's ability to understand "
    "context across long conversations and adapt its communication style represents a major "
    "advancement in human-AI interaction. However, experts emphasize the importance of responsible "
    "deployment, transparent documentation of capabilities and limitations, and ongoing research "
    "into AI safety and alignment with human values.",
    
    # Hawaii - Mauna Kea telescope discovery
    "Astronomers using the powerful telescopes atop Mauna Kea on Hawaii's Big Island have "
    "discovered an Earth-sized exoplanet in the habitable zone of a nearby star system. The "
    "planet, located 40 light-years away in the constellation Virgo, orbits within the "
    "temperature range where liquid water could exist on its surface. This groundbreaking "
    "discovery utilized the combined capabilities of the Keck Observatory and Subaru Telescope, "
    "both situated at the 13,796-foot summit of Mauna Kea, one of the world's premier "
    "astronomical research sites. The planet completes one orbit around its sun-like star "
    "every 287 days and receives similar amounts of stellar radiation as Earth. Scientists "
    "detected the planet using the radial velocity method, measuring tiny wobbles in the "
    "host star caused by the planet's gravitational pull. Initial spectroscopic analysis "
    "suggests the presence of an atmosphere, though its composition remains unknown. The "
    "discovery team includes researchers from the University of Hawaii's Institute for "
    "Astronomy and international collaborators. This finding adds to Hawaii's rich legacy "
    "of astronomical discoveries and reinforces Mauna Kea's critical role in advancing our "
    "understanding of potentially habitable worlds beyond our solar system.",
    
    # Hawaii - Renewable energy transition
    "Hawaii has reached a historic milestone in its transition to renewable energy, with "
    "solar and wind power now providing 65 percent of the state's electricity generation. "
    "This achievement places Hawaii well ahead of its ambitious goal to achieve 100 percent "
    "renewable energy by 2045, making it the first U.S. state with such a comprehensive "
    "clean energy mandate. The Big Island leads the transformation with 80 percent renewable "
    "penetration, primarily from geothermal energy tapped from Kilauea volcano's heat and "
    "extensive solar farms across Kona and Hilo districts. Kauai has successfully implemented "
    "innovative battery storage systems paired with massive solar arrays, enabling the island "
    "to run entirely on renewable energy during daylight hours. Oahu's grid integration "
    "challenges are being addressed through smart grid technology and distributed energy "
    "resources including rooftop solar installations on over 100,000 homes. Maui is pioneering "
    "offshore wind development projects that could provide stable baseload power. The transition "
    "has created over 15,000 clean energy jobs statewide while reducing the state's dependence "
    "on imported fossil fuels. However, challenges remain including grid stability management, "
    "energy storage capacity expansion, and ensuring affordable electricity rates for residents. "
    "Hawaii's success serves as a model for island nations worldwide seeking energy independence.",
    
    # Hawaii - Coral reef restoration
    "Marine biologists in Hawaii have achieved a breakthrough in coral reef restoration using "
    "innovative 3D-printed reef structures combined with heat-resistant coral strains. The "
    "project, led by researchers from the University of Hawaii at Manoa and the Hawaii Institute "
    "of Marine Biology, focuses on restoring damaged reefs around Oahu, Maui, and the Big Island "
    "that have suffered from coral bleaching events exacerbated by rising ocean temperatures. "
    "Scientists are cultivating coral species that demonstrate higher thermal tolerance, including "
    "varieties native to naturally warmer Hawaiian waters around volcanic vents. These resilient "
    "corals are grown in underwater nurseries before being transplanted onto artificial reef "
    "structures designed to mimic natural reef architecture. Early results show 75 percent "
    "survival rates and faster growth compared to traditional restoration methods. The reefs "
    "support critical ecosystems providing habitat for over 7,000 marine species, many endemic "
    "to Hawaii. The project also involves Native Hawaiian communities, incorporating traditional "
    "ecological knowledge and cultural practices into conservation efforts. Researchers are "
    "training local volunteers in coral propagation techniques and establishing monitoring "
    "programs across the islands. This work is crucial for protecting Hawaii's $800 million "
    "annual reef-related tourism economy while preserving these vital ecosystems for future "
    "generations and maintaining the cultural significance reefs hold in Hawaiian traditions.",
    
    # Hawaii - Volcanic monitoring
    "Scientists at the Hawaiian Volcano Observatory have deployed cutting-edge sensor networks "
    "across Kilauea and Mauna Loa volcanoes, dramatically improving eruption prediction "
    "capabilities on the Big Island. The new system integrates real-time seismic monitoring, "
    "ground deformation measurements using GPS and satellite radar, gas emission analysis, "
    "and thermal imaging from drones and satellites. This comprehensive approach allows "
    "researchers to detect subtle changes in volcanic activity days or even weeks before "
    "eruptions occur. Following the devastating 2018 Kilauea eruption that destroyed over "
    "700 homes in lower Puna, upgraded monitoring has become critical for community safety. "
    "The observatory now provides detailed hazard assessments and early warnings to Hawaii "
    "County Civil Defense, enabling timely evacuations and emergency response planning. "
    "Advanced machine learning algorithms analyze patterns in volcanic tremor signals to "
    "distinguish between harmless activity and precursors to major eruptions. The system "
    "successfully predicted recent lava lake activity in Kilauea's summit crater, demonstrating "
    "its effectiveness. Research findings are shared internationally, benefiting volcano "
    "monitoring programs worldwide. The observatory collaborates with USGS, university researchers, "
    "and local communities to ensure that scientific data translates into actionable safety "
    "measures. This enhanced monitoring represents a significant advancement in protecting "
    "the 200,000 residents living in volcano hazard zones across Hawaii Island.",
    
    # Hawaii - Indigenous language revitalization
    "Hawaii's indigenous language revitalization program has achieved remarkable success, with "
    "Hawaiian language immersion schools now serving over 3,000 students across the islands. "
    "This represents a dramatic recovery for a language that was nearly extinct just 50 years "
    "ago when fewer than 50 children spoke Hawaiian as their primary language. The program, "
    "known as Kaiapuni, provides complete K-12 education in Hawaiian, covering all subjects "
    "from mathematics and science to history and literature. Schools operate on all major "
    "islands including facilities on the Big Island in Keaau and Honokaa, on Maui in Lahaina, "
    "on Kauai in Waimea, and multiple schools on Oahu. Students consistently outperform their "
    "peers in standardized testing while maintaining fluency in both Hawaiian and English. "
    "The University of Hawaii system now offers undergraduate and graduate degrees taught "
    "entirely in Hawaiian, ensuring language transmission continues through higher education. "
    "Technology has played a crucial role with Hawaiian language apps, online dictionaries, "
    "and social media content helping to normalize daily Hawaiian usage among younger generations. "
    "The success has inspired indigenous language revitalization efforts globally, with "
    "delegations from Maori, Sami, and Native American communities visiting Hawaii to learn "
    "from the program. This cultural renaissance strengthens Hawaiian identity while preserving "
    "irreplaceable traditional knowledge encoded in the language's rich vocabulary and oral traditions.",
    
    # Hawaii - Sustainable agriculture
    "Hawaii's agricultural sector is undergoing a transformation toward food sustainability with "
    "innovative farming practices reducing the state's dependence on imported food. Currently, "
    "Hawaii imports over 85 percent of its food, but new initiatives are reversing this trend "
    "through vertical farming, aquaponics, and regenerative agriculture techniques. On the Big "
    "Island, farms in the Hamakua Coast and Waimea regions are producing specialty crops including "
    "coffee, macadamia nuts, and tropical fruits using sustainable methods that restore soil "
    "health and biodiversity. Kauai's agricultural parks feature innovative hydroponic operations "
    "growing leafy greens year-round with 90 percent less water than traditional farming. Maui's "
    "upcountry region has expanded production of vegetables, herbs, and flowers while implementing "
    "water conservation systems crucial during drought periods. The state has invested in agricultural "
    "infrastructure including processing facilities, distribution networks, and cold storage to "
    "support local farmers. Farm-to-school programs now source 25 percent of public school food "
    "locally, exposing students to fresh Hawaii-grown produce while supporting local economies. "
    "Young farmers are entering agriculture through training programs, land lease initiatives, "
    "and mentorship with experienced growers. These efforts address food security concerns, create "
    "rural employment opportunities, reduce carbon emissions from food transportation, and reconnect "
    "Hawaii residents with the islands' rich agricultural heritage and the traditional Hawaiian "
    "value of malama aina - caring for the land.",
    
    # Hawaii - Ocean conservation
    "Hawaii has established the largest fully protected marine conservation area in the United "
    "States, encompassing 582,578 square miles of ocean surrounding the Northwestern Hawaiian "
    "Islands. The Papahanaumokuakea Marine National Monument protects pristine coral reefs, "
    "endangered Hawaiian monk seals, green sea turtles, and nesting seabirds found nowhere "
    "else on Earth. Recent expansions have quadrupled the protected area, prohibiting commercial "
    "fishing and mineral extraction while allowing Native Hawaiian cultural practices and "
    "scientific research. Marine scientists from the University of Hawaii conduct regular "
    "expeditions documenting ecosystem health and discovering new species in these remote waters. "
    "The conservation area serves as a genetic reservoir for depleted fish populations around "
    "the main Hawaiian Islands, with larvae drifting south to replenish coastal reefs. Strict "
    "protections have allowed shark populations to recover, restoring balance to marine ecosystems. "
    "The monument also holds immense cultural significance, containing sacred sites and artifacts "
    "from ancient Polynesian voyagers who navigated these waters centuries ago. Tourism is "
    "carefully managed through limited permitted visits that support conservation funding while "
    "minimizing environmental impact. This conservation success demonstrates Hawaii's commitment "
    "to ocean stewardship and provides a blueprint for marine protected areas globally. The "
    "protected waters support sustainable fishing industries in adjacent areas while safeguarding "
    "biodiversity for future generations in the face of climate change and ocean acidification threats.",
]


# ============================================================================
# MODAL INFRUSTRUCTURE SETUP
# ============================================================================

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import modal

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    # For WattsOnAI
    .uv_pip_install( 
        "sentencepiece", 
        "accelerate",
        "evaluate", 
        "rouge_score", 
        "nltk", 
        "transformers",
        "datasets<4.0.0",
        "huggingface_hub", 
        "tensorboard",
        "flask", 
        "plotly",
        "dash",
        "requests",
        "torch", 
        "torchvision",
        "psutil",  # For CPU/memory monitoring
        "nvidia-ml-py3",   # NVIDIA GPU monitoring
        "pandas",  # Often needed for CSV output
        "numpy",
        "matplotlib",  # For drawing plots
        "seaborn",  # For better plots
        "mysql-connector-python", 
        "python-dotenv"
    )
    .env({"HF_HOME": "/model_cache"})
    .add_local_python_source("WattsOnAI")  # Add the local Python source
)

app = modal.App("finetune", image=train_image)


# ============================================================================
# GPU & MODEL SETUP
# ============================================================================

# TODO: Utilize other GPUs
GPU_TYPE = "A100-40GB"
REGION = "us-west-1"
TIMEOUT_HOURS = 5
MAX_RETRIES = 3

VOL_MOUNT_PATH = Path("/vol")
MODEL_CACHE_PATH = Path("/model_cache")
CHECKPOINT_PATH = Path("/checkpoints")
DATASET_CACHE_PATH = Path("/dataset_cache_volume")

# ============================================================================
# VOLUME CONFIGURATION
# ============================================================================

output = modal.Volume.from_name(
    "output", create_if_missing=True
)

# Add model-specific restart tracking
restart_tracker_dict = modal.Dict.from_name(
    "finetune-restart-tracker", create_if_missing=True
)

# ============================================================================
# HANDLING PREMPTIONS
# A preemption event in training models is the temporary or permanent termination of a training job by an external scheduler, most commonly a cloud provider. 
# ============================================================================

def track_restarts(restart_dict, model_name: str):
    """Track the number of times this specific model has been restarted due to preemption"""
    key = f"restart_count_{model_name.replace('/', '_').replace('-', '_')}"
    current_count = restart_dict.get(key, 0)
    restart_dict[key] = current_count + 1
    print(f"Model {model_name} restart count: {current_count}")
    return current_count

# ============================================================================
# FINE-TUNING ON XSum-Dataset for both models
# ============================================================================

@app.cls(
    gpu=GPU_TYPE, 
    cpu=(2.0, 6.0),  # Request 4 CPU cores explicitly
    timeout=TIMEOUT_HOURS * 3600,  # Add this line - converts hours to seconds
    scaledown_window=60 * 5, 
    enable_memory_snapshot=True,
    image=train_image,
    volumes={
        VOL_MOUNT_PATH: output
    },
    region=REGION,
    secrets=[modal.Secret.from_dotenv()]  
)
class GPUMonitoringClass:
    @modal.enter()
    def setup(self):
        try:
            import os
            from huggingface_hub import login
            from WattsOnAI import monitor
            print("WattsOnAI imported successfully")
            
            # Default: AWS California
            # Initialize instance variables
            self.cloud_region = os.environ.get('MODAL_REGION', 'us-west-1') # 
            self.task_id = os.environ.get('MODAL_TASK_ID', 'Unknown')
            self.modal_workspace = os.environ.get('MODAL_WORKSPACE_NAME', 'Unknown')
            
            # Print Modal Environement
            print(f"\n{'='*70}")
            print(f"🌍 Modal Container Environment")
            print(f"{'='*70}")
            print(f"📍 Modal Cloud Region: {self.cloud_region}")
            
            # Print other useful Modal environment variables
            print(f"🔧 Task ID: {self.task_id}")
            print(f"🏢 Workspace: {self.modal_workspace}")
            print(f"{'='*70}\n")
            
            # Try to get token from environment (Modal secret)
            # Login/Authentication into HuggingFace
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                try:
                    # Note: Suppress the message `Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.`
                    # login(token=hf_token)
                    print("✅ Successfully logged into Hugging Face Hub")             
                    
                except Exception as login_error:
                    print(f"⚠️ HF login failed: {login_error}")
                    print("Proceeding without authentication - may fail on private repos")
            else:
                print("⚠️ No HF_TOKEN found in environment variables")
                print("Make sure your .env file contains: HF_TOKEN=your_token_here")        
            
        except ImportError as e:
            print(f"Import error: {e}")
            
    @modal.method()
    def print_environment_info(self):
        """Print detailed Modal environment information"""
        import os
        import json
        
        env_info = {
            "Modal Cloud Region": os.environ.get('MODAL_REGION', 'Unknown'),
            "Cloud Provider": os.environ.get('MODAL_CLOUD_PROVIDER', 'Unknown'),
            "Task ID": os.environ.get('MODAL_TASK_ID', 'Unknown'),
            "Function Name": os.environ.get('MODAL_FUNCTION_NAME', 'Unknown'),
            "Workspace": os.environ.get('MODAL_WORKSPACE_NAME', 'Unknown'),
            "Environment": os.environ.get('MODAL_ENVIRONMENT', 'Unknown'),
        }
        
        print(f"\n{'='*70}")
        print(f"🔍 Modal Container Environment Details")
        print(f"{'='*70}")
        for key, value in env_info.items():
            print(f"  {key:<20}: {value}")
        print(f"{'='*70}\n")
        
    @modal.method()    
    def finetune(
        self,
        num_train_epochs: int = 1, 
        size_percentage: int = 10,
        model_name: str = None,
    ):
        import evaluate
        import numpy as np
        import torch
        from datasets import load_dataset
        from datasets import load_dataset_builder # datasets => 4.0.0.
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
        
        if model_name is None:
            raise ValueError("`model_name` parameter cannot be None")
        
        # Configuration
        padding_token_id = -100
        batch_size = 8
        cloud_region = self.cloud_region
        restarts = track_restarts(restart_tracker_dict, model_name)
        model_safe_name = model_name.replace('/', '_').replace('-', '_')
        
        # ============================================================
        # PHASE 1: Device Detection & Configuration
        # ============================================================
        print(f"\n{'='*70}")
        print(f"🖥️  PHASE 1: Device Configuration")
        print(f"{'='*70}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cpu_device = torch.device("cpu")
        
        print(f"Primary device: {device}")
        print(f"CPU cores available: {torch.get_num_threads()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"{'='*70}\n")
        
        # ============================================================
        # PHASE 2: Check for Existing Model
        # ============================================================
        print(f"\n{'='*70}")
        print(f"🔍 PHASE 2: Model Existence Check")
        print(f"{'='*70}")
        
        final_model_path = VOL_MOUNT_PATH / f"{model_safe_name}_{cloud_region}_final_model"
        
        if final_model_path.exists():
            print(f"⚠️ Final model for {model_name} already exists at {final_model_path}")
            print("Skipping training. Delete the directory if you want to retrain.")
            return None
        print(f"✅ No existing model found. Proceeding with training.")
        print(f"{'='*70}\n")
        
        # ============================================================
        # PHASE 3: CPU - Data Loading & Preprocessing
        # ============================================================
        print(f"\n{'='*70}")
        print(f"🔄 PHASE 3: CPU - Data Loading & Preprocessing")
        print(f"💻 Device: CPU")
        print(f"{'='*70}")
        
        print("📚 Loading dataset on CPU...")
        if size_percentage:
            xsum_train = load_dataset("EdinburghNLP/xsum", split=f"train[:{size_percentage}%]")
            xsum_test = load_dataset("EdinburghNLP/xsum", split=f"test[:{size_percentage}%]")
        else:
            xsum_train = load_dataset("EdinburghNLP/xsum", split="train")
            xsum_test = load_dataset("EdinburghNLP/xsum", split="test")
            
        print(f"✅ Loaded {len(xsum_train)} training samples and {len(xsum_test)} test samples")
        print(f"{'='*70}\n")
        
        # ============================================================
        # PHASE 4: GPU/CPU - Model Loading
        # ============================================================
        print(f"\n{'='*70}")
        print(f"📦 PHASE 4: Model Loading")
        print(f"🎯 Target Device: {device}")
        print(f"{'='*70}")
        
        if model_name in SMALL_MODELS:
            model_config = SMALL_MODELS[model_name]
            tokenizer = AutoTokenizer.from_pretrained(
                model_config['model_name'],
                ignore_mismatched_sizes=True,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_config['model_name'],
                ignore_mismatched_sizes=True, 
                low_cpu_mem_usage=True 
            )
            
            if hasattr(model, 'generation_config'):
                model.generation_config.early_stopping = True
                model.generation_config.num_beams = 4
                model.generation_config.no_repeat_ngram_size = 3
                if tokenizer.bos_token_id is not None:
                    model.generation_config.forced_bos_token_id = tokenizer.bos_token_id
                print("✅ Generation config updated")
            
            print(f"🔄 Moving model to {device}...")
            model = model.to(device)
            
            print(f"✅ Model loaded on {device}")
            print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"   Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9:.2f} GB")
            
            if torch.cuda.is_available():
                print(f"   GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"   GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        print(f"{'='*70}\n")
        
        # ============================================================
        # PHASE 5: CPU - ROUGE Metric Setup
        # ============================================================
        print(f"\n{'='*70}")
        print(f"📊 PHASE 5: CPU - ROUGE Metric Setup")
        print(f"💻 Device: CPU (Metric computation)")
        print(f"{'='*70}")
        
        print("📊 Loading ROUGE metric...")
        rouge = evaluate.load("rouge")
        
        def compute_metrics(eval_pred): 
            """Metric computation runs on CPU"""
            predictions, labels = eval_pred
            
            # Move from GPU to CPU if needed
            if torch.is_tensor(predictions):
                predictions = predictions.cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            
            # Decode predictions 
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
            # Replace -100 in labels with pad_token_id
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Compute ROUGE scores
            result = rouge.compute(
                predictions=decoded_preds, 
                references=decoded_labels, 
                use_stemmer=True
            )
            
            # Calculate average generation length
            prediction_lens = [
                np.count_nonzero(pred != tokenizer.pad_token_id)
                for pred in predictions
            ]
            result["gen_len"] = np.mean(prediction_lens)
            
            return {k: round(v, 4) for k, v in result.items()}
        
        print(f"✅ ROUGE metric configured")
        print(f"{'='*70}\n")

        # ============================================================
        # PHASE 6: CPU - Data Tokenization & Batching
        # ============================================================
        print(f"\n{'='*70}")
        print(f"📦 PHASE 6: CPU - Data Tokenization & Batching")
        print(f"💻 Device: CPU (Batch preparation)")
        print(f"{'='*70}")

        import psutil
        num_cpus = psutil.cpu_count(logical=False) or 4
        print(f"🔄 Tokenizing with {num_cpus} CPU cores...")
        
        def preprocess(batch):
            # Different prefix for different models
            if 't5' in model_name.lower() or 'flan' in model_name.lower():
                # T5/FLAN-T5 needs "summarize:" prefix
                inputs = ["summarize: " + doc for doc in batch["document"]]
            else:
                # BART doesn't need prefix
                inputs = batch["document"]
            
            model_inputs = tokenizer(
                inputs, max_length=512, truncation=True, padding="max_length"
            )
            labels = tokenizer(
                text_target=batch["summary"],
                max_length=128,
                truncation=True,
                padding="max_length",
            )
            labels["input_ids"] = [
                [l if l != tokenizer.pad_token_id else padding_token_id for l in label]
                for label in labels["input_ids"]
            ]
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_xsum_train = xsum_train.map(
            preprocess, 
            batched=True, 
            remove_columns=["document", "summary", "id"],
            num_proc=num_cpus
        )
        tokenized_xsum_test = xsum_test.map(
            preprocess, 
            batched=True, 
            remove_columns=["document", "summary", "id"],
            num_proc=num_cpus
        )
        
        print(f"✅ Tokenization complete (CPU-based)")
    
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=padding_token_id,
            pad_to_multiple_of=batch_size,
        )
        
        print("✅ Data collator configured")
        print(f"{'='*70}\n")
        
        # ============================================================
        # PHASE 7: GPU - Model Training
        # ============================================================
        print(f"\n{'='*70}")
        print(f"🚀 PHASE 7: GPU - Model Training")
        print(f"🎮 Device: {device}")
        print(f"💻 CPU workers for data loading: 4")
        print(f"{'='*70}\n")

        # Reference: https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(VOL_MOUNT_PATH / f"model_{model_safe_name}_{cloud_region}"),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            predict_with_generate=True,
            learning_rate=3e-5,
            num_train_epochs=num_train_epochs,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            
            eval_strategy="steps", 
            eval_steps=500, 
            logging_strategy="steps",
            logging_steps=500,
            save_strategy="steps",
            save_steps=500,
            
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            greater_is_better=True,
            resume_from_checkpoint=None,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_xsum_train,
            eval_dataset=tokenized_xsum_test,
            compute_metrics=compute_metrics,
        )

        # Checkpoint handling
        try:
            checkpoint_dir = VOL_MOUNT_PATH / f"model_{model_safe_name}_{cloud_region}"
            resume_checkpoint = None
            
            if restarts > 0:
                import glob
                checkpoint_pattern = str(checkpoint_dir / "checkpoint-*")
                checkpoints = glob.glob(checkpoint_pattern)
                
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
                    trainer_state_path = Path(latest_checkpoint) / "trainer_state.json"
                    
                    if trainer_state_path.exists():
                        resume_checkpoint = latest_checkpoint
                        print(f"Resuming {model_name} from checkpoint: {resume_checkpoint}")
                    else:
                        print(f"Checkpoint {latest_checkpoint} is incomplete, starting {model_name} fresh")
                else:
                    print(f"No valid checkpoints found for {model_name}, starting fresh")
            
            # Train the model
            trainer.train(resume_from_checkpoint=resume_checkpoint)
            
            # Final evaluation
            print(f"\n{'='*70}")
            print(f"📊 Running Final Evaluation for {model_name}")
            print(f"{'='*70}")
            
            eval_results = trainer.evaluate()
            
            print(f"\nEvaluation Results:")
            for metric_name, metric_value in eval_results.items():
                print(f"  {metric_name}: {metric_value}")
            print(f"{'='*70}\n")
            
        except KeyboardInterrupt:
            print("received interrupt; saving state and model")
            trainer.save_state()
            trainer.save_model()
            raise
        
        # ============================================================
        # PHASE 8: CPU - Model Saving
        # ============================================================
        print(f"\n{'='*70}")
        print(f"💾 PHASE 8: CPU - Model Saving")
        print(f"💻 Device: CPU (Disk I/O)")
        print(f"{'='*70}")
        
        final_model_path = VOL_MOUNT_PATH / f"{model_safe_name}_{cloud_region}_final_model"
        final_tokenizer_path = VOL_MOUNT_PATH / f"{model_safe_name}_{cloud_region}_final_tokenizer"
        
        print("🔄 Moving model to CPU for saving...")
        model = model.cpu()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"✅ GPU memory freed")
        
        model.save_pretrained(str(final_model_path))
        tokenizer.save_pretrained(str(final_tokenizer_path))
        output.commit()
        
        # Reset restart counter
        restart_tracker_dict[f"restart_count_{model_safe_name}_{cloud_region}"] = 0

        print(f"✅ {model_name} model training completed and saved to {final_model_path}")
        print(f"{'='*70}\n")
        
        return eval_results

    @modal.method()
    def inference(
        self, 
        model_name: str, 
        test_samples: int = 10
    ): 
        from transformers import pipeline, AutoTokenizer
        import torch
        import time
        import numpy as np
        
        print(f"\n{'='*70}")
        print(f"💻 Model Inference with Hugging-Face Pipeline")
        print(f"{'='*70}")
            
        cloud_region = self.cloud_region
        model_safe_name = model_name.replace('/', '_').replace('-', '_')
    
        # Load the fine-tuned model from Modal volume
        final_model_path = VOL_MOUNT_PATH / f"{model_safe_name}_{cloud_region}_final_model"
        final_tokenizer_path = VOL_MOUNT_PATH / f"{model_safe_name}_{cloud_region}_final_tokenizer"
        
        if not final_model_path.exists():
            print(f"⚠️ Model not found at {final_model_path}")
            print("Please train the model first")
            raise FileNotFoundError(f"Model not found: {final_model_path}")
            
        
        if not final_tokenizer_path.exists():
            print(f"⚠️ Tokenizer not found at {final_tokenizer_path}")
            print("Please specify the tokenizer first")
            raise FileNotFoundError(f"Tokenizer not found: {final_tokenizer_path}")        

        print(f"📥 Loading model from: {final_model_path}")
        print(f"📥 Loading tokenizer from: {final_tokenizer_path}")


        tokenizer = AutoTokenizer.from_pretrained(
            str(final_tokenizer_path),
            # use_fast=False  # ✅ Force slow tokenizer to avoid conversion errors
        )
        
        print(f"📥 Loading tokenizer from: {final_tokenizer_path}")
        
        # ============================================================
        # Option 1: device="cpu" (Explicit string)
        # ============================================================
        print(f"🔧 Creating summarization pipeline on CPU...")
                    
        summarizer = pipeline(
            "summarization",
            model=str(final_model_path),
            tokenizer=tokenizer, 
            device="cpu",  # ✅ Use CPU explicitly
            batch_size=8    # Process 8 samples at once
        )
        
        print(f"✅ Pipeline created on CPU")
        print(f"   Device: {summarizer.device}")
        print(f"   Model: {summarizer.model.__class__.__name__}")
        print(f"   Tokenizer: {summarizer.tokenizer.__class__.__name__}")
        
        # ============================================================
        # Prepare test data
        # ============================================================
        print(f"\nPreparing {test_samples} test samples...")
        
        # Prepare test articles based on model type
        if 't5' in model_name.lower() or 'flan' in model_name.lower():
            test_articles_formatted = ["summarize: " + article for article in TEST_ARTICLES]
            model_type = "T5"
        else:
            test_articles_formatted = TEST_ARTICLES
            model_type = "BART"
            
        # Repeat to create test set
        test_set = (test_articles_formatted * (test_samples // len(test_articles_formatted) + 1))[:test_samples]
        
        input_token_counts = []
        input_char_counts = []
        
        for text in test_set: 
            # Remove prefix for counting
            text_clean = text.replace("summarize: ", "")
            input_char_counts.append(len(text_clean))
            
            # Tokenize to count tokens
            tokens = tokenizer.encode(text, add_special_tokens = True)
            input_token_counts.append(len(tokens))
        
        # ============================================================
        # Run CPU inference and benchmark
        # ============================================================
        print(f"\n🚀 Running CPU inference on {len(test_set)} samples...")
        
        start_time = time.time()
    
        # Run inference
        summaries = summarizer(
            test_set,
            max_length=128,
            min_length=30,
            do_sample=False,
            batch_size=8  # Process in batches for efficiency
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # ============================================================
        # Calculate compression metrics
        # ============================================================
        output_token_counts = []
        output_char_counts = []
        
        for summary in summaries:
            summary_text = summary['summary_text']
            output_char_counts.append(len(summary_text))
            
            # Tokenize to count tokens
            tokens = tokenizer.encode(summary_text, add_special_tokens=True)
            output_token_counts.append(len(tokens))
        
        # Calculate compression ratios
        avg_input_tokens = np.mean(input_token_counts)
        avg_output_tokens = np.mean(output_token_counts)
        avg_input_chars = np.mean(input_char_counts)
        avg_output_chars = np.mean(output_char_counts)
        
        token_compression_ratio = avg_input_tokens / avg_output_tokens if avg_output_tokens > 0 else 0
        char_compression_ratio = avg_input_chars / avg_output_chars if avg_output_chars > 0 else 0
        
        # ============================================================
        # Calculate metrics
        # ============================================================
        results = {
            "model": model_name,
            "region": cloud_region,
            "device": "cpu",
            "num_samples": len(test_set),
            "total_time_seconds": elapsed_time,
            "avg_time_per_sample": elapsed_time / len(test_set),
            "throughput_samples_per_second": len(test_set) / elapsed_time,
            "compression_metrics": {
                "avg_input_tokens": avg_input_tokens,
                "avg_output_tokens": avg_output_tokens,
                "avg_input_chars": avg_input_chars,
                "avg_output_chars": avg_output_chars,
                "token_compression_ratio": token_compression_ratio,
                "char_compression_ratio": char_compression_ratio
            },
            "sample_summaries": []
        }
        
        # Store sample summaries
        for idx, (article, summary) in enumerate(zip(test_set[:5], summaries[:5])):
            results["sample_summaries"].append({
                "article_preview": article[:100] + "...",
                "summary": summary['summary_text']
            })
            
        # ============================================================
        # Print results
        # ============================================================
        print(f"\n{'='*70}")
        print(f"📊 CPU Inference Results")
        print(f"{'='*70}")
        print(f"  Model: {model_name}")
        print(f"  Region: {cloud_region}")
        print(f"  Device: CPU")
        print(f"  Samples processed: {results['num_samples']}")
        print(f"  Total time: {results['total_time_seconds']:.2f}s")
        print(f"  Avg time/sample: {results['avg_time_per_sample']:.3f}s")
        print(f"  Throughput: {results['throughput_samples_per_second']:.2f} samples/sec")
        print(f"\n📏 Compression Metrics:")
        print(f"  Avg input tokens: {avg_input_tokens:.1f}")
        print(f"  Avg output tokens: {avg_output_tokens:.1f}")
        print(f"  Token compression ratio: {token_compression_ratio:.2f}x")
        print(f"  Avg input chars: {avg_input_chars:.1f}")
        print(f"  Avg output chars: {avg_output_chars:.1f}")
        print(f"  Char compression ratio: {char_compression_ratio:.2f}x")
        print(f"{'='*70}\n")
        
        # Print sample summaries
        print("📝 Sample Summaries:")
        for idx, sample in enumerate(results["sample_summaries"], 1):
            print(f"\n  {idx}. Original: {sample['article_preview']}")
            print(f"     Summary: {sample['summary']}\n")
            
        print(f"✅ {model_name} model inference completed")
        
        return results
            
    def save_to_modal_volume(self, csv_files, json_files):
        """Save CSV and JSON files to Modal volume with organized structure"""
        import json
        
        try:
            monitoring_dir = VOL_MOUNT_PATH / "monitoring_data"
            csv_dir = monitoring_dir / "csv"
            json_dir = monitoring_dir / "json"
            
            csv_dir.mkdir(parents=True, exist_ok=True)
            json_dir.mkdir(parents=True, exist_ok=True)
            
            saved_file_paths = {"csv": [], "json": []}
            
            # Save CSV files
            for filename, content in csv_files.items():
                file_path = csv_dir / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                saved_file_paths["csv"].append(str(file_path))
                print(f"✓ Saved CSV: {file_path}")
            
            # Save JSON files
            for filename, content in json_files.items():
                file_path = json_dir / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    if isinstance(content, str):
                        f.write(content)
                    else:
                        json.dump(content, f, indent=2, ensure_ascii=False)
                saved_file_paths["json"].append(str(file_path))
                print(f"✓ Saved JSON: {file_path}")
                       
            output.commit()
            print(f"✓ All monitoring data committed to volume at {monitoring_dir}")
            
        except Exception as e:
            print(f"❌ Error saving to volume: {e}")
            raise    

    @modal.method()
    def monitor_workload(self, models_to_train: Optional[list] = None):
        """Run monitoring and finetune models, returning CSV file paths and performance JSON data"""
        if models_to_train is None:
            models_to_train = ["bart-small", "flan-t5-small"]
        
        csv_files = {}
        json_files = {}
        
        try:
            from WattsOnAI import monitor
            import pandas as pd
            import glob
            import json

            print(f"\n{'='*60}")
            print(f"Processing region: {self.cloud_region}")
            print(f"{'='*60}\n")
            
            for model_name in models_to_train:
                model_safe_name = model_name.replace('/', '_').replace('-', '_')
                
                # Start monitoring
                task_name = f"workload_{model_safe_name}_{self.cloud_region}"
                monitor.start(
                    task_name=task_name,
                    sampling_interval=1,
                    output_format="csv",
                    cloud_region=self.cloud_region, 
                    additional_metrics=["CPU", "DRAM"]
                )
                
                # Finetune Model
                evaluation_json = self.finetune.local(model_name=model_name)
                
                # Sample Inference in Finetuned Model
                inference_json = self.inference.local(model_name=model_name)
                
                # Stop monitoring
                performance_json = monitor.stop()
                
                # Extract region
                region = 'Unknown_Region'
                if performance_json:
                    try:
                        metadata = performance_json.get('metadata', {})
                        if metadata and isinstance(metadata, dict):
                            region = metadata.get('cloud_region', 'Unknown_Region')
                        print(f"✓ Extracted region for {model_name}: {region}")
                    except Exception as e:
                        print(f"⚠️ Error extracting region for {model_name}: {e}")
         
                # Store JSON files
                if performance_json:
                    json_filename = f"{model_safe_name}_{region}_performance.json"
                    json_content = json.dumps(performance_json, indent=2, ensure_ascii=False)
                    json_files[json_filename] = json_content
                    print(f"Stored JSON performance data for {model_name} trained in {region}")
                    
                if inference_json:
                    json_filename = f"{model_safe_name}_{region}_inference.json"
                    json_content = json.dumps(inference_json, indent=2, ensure_ascii=False)
                    json_files[json_filename] = json_content
                    print(f"Stored JSON inference data for {model_name} trained in {region}")
                    
                if evaluation_json:
                    json_filename = f"{model_safe_name}_{region}_evaluation.json"
                    json_content = json.dumps(evaluation_json, indent=2, ensure_ascii=False)
                    json_files[json_filename] = json_content
                    print(f"Stored JSON evaluation data for {model_name} trained in {region}")

                # Store CSV files
                csv_pattern = f'/root/{task_name}_*.csv'
                model_csv_files = glob.glob(csv_pattern)

                if model_csv_files:
                    csv_path = model_csv_files[0]
                    df = pd.read_csv(csv_path)
                    csv_filename = f"{model_safe_name}_{region}_monitor.csv"
                    csv_files[csv_filename] = df.to_csv(index=False)
                    print(f"Stored CSV content for {model_name}")
            
            # Save to Modal volume
            if csv_files and json_files:
                self.save_to_modal_volume(csv_files, json_files)
                print(f"✅ Successfully saved {len(csv_files)} CSV and {len(json_files)} JSON files to volume")
            else:
                print("⚠️ No files to save to volume")
                
            print(f"📊 Returning {len(csv_files)} CSV files and {len(json_files)} JSON files")
            return csv_files, json_files
              
        except Exception as e:
            print(f"Error during monitoring and training: {e}")
            return csv_files, json_files
    
    
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Function to reset the restart tracker
@app.function()
def reset_restart_tracker():
    """Reset the restart tracker count"""
    restart_tracker_dict.clear()
    print("✅ Restart tracker cleared")

@app.function(
    volumes={VOL_MOUNT_PATH: output},
    timeout=300  # 5 minutes timeout
)
def clear_volume_contents():
    """
    Delete ALL files and subdirectories in the entire volume.
    This clears everything in /vol/
    
    Returns:
        dict: Summary of deletion operation
    """
    import shutil
    from pathlib import Path
    
    deleted_count = 0
    errors = []
    
    try:
        # Get the volume mount path
        volume_path = Path(VOL_MOUNT_PATH)
        
        if not volume_path.exists():
            return {
                "status": "warning",
                "message": f"Volume path {VOL_MOUNT_PATH} does not exist",
                "deleted_items": 0,
                "errors": []
            }
        
        print(f"🗑️  Starting to clear all contents in volume: {volume_path}")
        
        # Iterate through all items in the volume root
        for item in volume_path.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                    deleted_count += 1
                    print(f"✓ Deleted file: {item.name}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_count += 1
                    print(f"✓ Deleted directory: {item.name}")
            except Exception as e:
                error_msg = f"Failed to delete {item.name}: {e}"
                errors.append(error_msg)
                print(f"❌ {error_msg}")
        
        # Commit changes to persist the deletions
        output.commit()
        
        result = {
            "status": "success",
            "message": f"Cleared {deleted_count} items from volume",
            "deleted_items": deleted_count,
            "errors": errors
        }
        
        print(f"✅ Volume cleared: {deleted_count} items deleted")
        return result
        
    except Exception as e:
        result = {
            "status": "error",
            "message": f"Error clearing volume: {str(e)}",
            "deleted_items": deleted_count,
            "errors": errors
        }
        print(f"❌ Error: {e}")
        return result

# modal run unsloth_finetune.py::reset
@app.local_entrypoint()
def reset():
    """Reset the preemption counter"""
    reset_restart_tracker.remote()
    print("✅ Success: Restart counter has been reset")
    
    print("🗑️  Clearing ALL volume contents...")
    result = clear_volume_contents.remote()
    print(f"✅ Success: Cleanup result: {result}")


# ✅ Testing Satisfied
@app.local_entrypoint()
def test_carbon_regions():
    """Test carbon intensity lookup for different AWS/Azure regions"""
    import time
    # Fix to utilize regions
    from WattsOnAI.get_carbon_density import get_current_carbon_intensity
    
    # Regions used: AWS California, AZR Texas, abd GCP Oregon
    # Note: Texas doesn't have full AWS Region but has Local Zones in Dallas/Houston
    # Reference: https://www.thisoldhouse.com/solar-alternative-energy/renewable-energy-by-state
    regions = [
        {
            "name": "California",
            "cloud_region": 'us-west-1'
        },
        {
            "name": "Oregon",
            "cloud_region": 'us-west-2'
         }
    ]
    
    print(f"\n{'='*70}")
    print(f"{'Carbon Intensity Region Testing':^70}")
    print(f"{'='*70}\n")
    
    results = []
    
    for region in regions:
        time.sleep(5)
        print(f"\n📍 Testing: {region['name']}")
        
        try:
            result = get_current_carbon_intensity(
                username="rkv_exploring", 
                password="", 
                cloud_region=region['cloud_region']
            )

            print(f"✅ Cloud Region: {result['cloud_region']}")
            print(f"✅ Carbon Intensity: {result['value']} {result['units']}")
            
            results.append({
                "location": region['name'],
                "watttime_region": result['cloud_region'],
                "intensity": result['value'],
                "units": result['units']
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"{'Test Summary':^70}")
    print(f"{'='*70}")
    for r in results:
        print(f"{r['location']:<20} → {r['watttime_region']:<20} ({r['intensity']:.2f} {r['units']})")
    print(f"{'='*70}\n")

# ✅ Testing Satisfied 
@app.local_entrypoint()
def test_gpu_class():
    """Test GPU class configuration without spawning containers"""
    
    print(f"\n{'='*70}")
    print(f"{'GPU Class Configuration Test (Lightweight)':^70}")
    print(f"{'='*70}\n")
    
    region_names = ["us-west", "us-east"]
    
    print(f"📋 Testing {len(region_names)} region configurations...\n")
    for idx, region in enumerate(region_names, 1):
        try:
            # Test class configuration (doesn't spawn container)
            RegionalGPUClass = GPUMonitoringClass.with_options(region=region)
            # Create instance and call the method
            gpu_instance = RegionalGPUClass()
            gpu_instance.print_environment_info.remote()
            print(f"   ✅ Configuration created successfully")
            
        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")
        print()  # Blank line between regions

@app.function(image=train_image)
def diagnose_monitoring():
    """Comprehensive diagnostic test"""
    print(f"\n{'='*70}")
    print(f"🔍 Monitoring Diagnostic Test")
    print(f"{'='*70}\n")
    
    # Test 1: psutil
    print("Test 1: psutil")
    try:
        import psutil
        print(f"   ✅ psutil version: {psutil.__version__}")
        print(f"   ✅ CPU count: {psutil.cpu_count()}")
        print(f"   ✅ CPU percent: {psutil.cpu_percent(interval=1)}%")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 2: WattsOnAI import
    print("\nTest 2: WattsOnAI import")
    try:
        from WattsOnAI import monitor
        print(f"   ✅ WattsOnAI imported")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Test 3: monitor.start/stop
    print("\nTest 3: monitor.start/stop")
    try:
        monitor.start(
            task_name="diagnostic_test",
            sampling_interval=1,
            output_format="csv",
            cloud_region="us-west-1"
        )
        print(f"   ✅ monitor.start() succeeded")
        
        import time
        time.sleep(5)  # Let it collect some data
        
        result = monitor.stop()
        print(f"   ✅ monitor.stop() succeeded")
        print(f"   Result type: {type(result)}")
        if result:
            print(f"   Keys: {list(result.keys())}")
        else:
            print(f"   ⚠️ Result is None")
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}\n")


@app.function(image=train_image)
def diagnose_cpu_monitoring():
    """Diagnose what CPU monitoring capabilities are available in Modal"""
    import psutil
    import os
    import platform
    
    print(f"\n{'='*70}")
    print(f"🔍 CPU Monitoring Diagnostic for Modal Container")
    print(f"{'='*70}\n")
    
    # ============================================================
    # System Information
    # ============================================================
    print("📊 System Information:")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Machine: {platform.machine()}")
    print(f"   Processor: {platform.processor()}")
    
    # ============================================================
    # CPU Information
    # ============================================================
    print(f"\n💻 CPU Information:")
    print(f"   Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"   Logical cores: {psutil.cpu_count(logical=True)}")
    
    try:
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            print(f"   CPU frequency: {cpu_freq.current:.2f} MHz")
            print(f"   Min frequency: {cpu_freq.min:.2f} MHz")
            print(f"   Max frequency: {cpu_freq.max:.2f} MHz")
    except Exception as e:
        print(f"   CPU frequency: Not available ({e})")
    
    # ============================================================
    # Test psutil CPU metrics
    # ============================================================
    print(f"\n📈 Testing psutil CPU metrics:")
    try:
        # Test with different intervals
        for interval in [0.05, 0.1, 0.5, 1.0]:
            cpu_percent = psutil.cpu_percent(interval=interval)
            print(f"   CPU usage (interval={interval}s): {cpu_percent}%")
        
        # Per-core usage
        per_core = psutil.cpu_percent(interval=1, percpu=True)
        print(f"   Per-core usage: {per_core}")
        
        # CPU times
        cpu_times = psutil.cpu_times()
        print(f"   CPU times: user={cpu_times.user:.2f}s, system={cpu_times.system:.2f}s")
        
        print(f"✅ psutil CPU monitoring: WORKING")
        
    except Exception as e:
        print(f"❌ psutil CPU monitoring: FAILED - {e}")
    
    # ============================================================
    # Check for powercap (RAPL)
    # ============================================================
    print(f"\n⚡ Checking for RAPL power monitoring:")
    powercap_path = "/sys/class/powercap"
    
    if os.path.exists(powercap_path):
        print(f"✅ {powercap_path} exists")
        
        try:
            entries = os.listdir(powercap_path)
            print(f"   Found entries: {entries}")
            
            # Check for intel-rapl
            intel_rapl = [e for e in entries if e.startswith("intel-rapl")]
            if intel_rapl:
                print(f"   ✅ Intel RAPL found: {intel_rapl}")
                
                # Try to read energy
                for rapl in intel_rapl:
                    if ":" not in rapl[len("intel-rapl:"):]:
                        energy_path = os.path.join(powercap_path, rapl, "energy_uj")
                        if os.path.exists(energy_path):
                            try:
                                with open(energy_path, "r") as f:
                                    energy = f.read().strip()
                                print(f"   ✅ Can read {rapl}/energy_uj: {energy} μJ")
                            except PermissionError:
                                print(f"   ❌ Permission denied reading {rapl}/energy_uj")
                            except Exception as e:
                                print(f"   ❌ Error reading {rapl}/energy_uj: {e}")
            else:
                print(f"   ⚠️ No Intel RAPL entries found")
                
        except PermissionError:
            print(f"❌ Permission denied accessing {powercap_path}")
        except Exception as e:
            print(f"❌ Error accessing {powercap_path}: {e}")
    else:
        print(f"❌ {powercap_path} does NOT exist")
        print(f"   RAPL power monitoring not available in this container")
    
    # ============================================================
    # Alternative: Check /proc/cpuinfo
    # ============================================================
    print(f"\n📄 Checking /proc/cpuinfo:")
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                lines = f.readlines()[:20]  # First 20 lines
            print(f"✅ /proc/cpuinfo accessible")
            print(f"   First few lines:")
            for line in lines[:5]:
                print(f"      {line.strip()}")
        except Exception as e:
            print(f"❌ Error reading /proc/cpuinfo: {e}")
    else:
        print(f"❌ /proc/cpuinfo does not exist")
    
    # ============================================================
    # Test memory monitoring
    # ============================================================
    print(f"\n🧠 Testing memory monitoring:")
    try:
        memory = psutil.virtual_memory()
        print(f"✅ Memory monitoring working:")
        print(f"   Total: {memory.total / 1e9:.2f} GB")
        print(f"   Available: {memory.available / 1e9:.2f} GB")
        print(f"   Used: {memory.used / 1e9:.2f} GB ({memory.percent}%)")
    except Exception as e:
        print(f"❌ Memory monitoring failed: {e}")
    
    print(f"\n{'='*70}\n")

# ============================================================================
# WSGI APP (Dashboard)
# ============================================================================

# Create file path patterns based on filters
def create_file_patterns(base_pattern, models=None, regions=None):
    """
    Create glob patterns for file matching
    
    Returns list of patterns like:
    - If both specified: ["bart_small_us_west_1_performance.json", ...]
    - If regions only: ["*_us_west_1_performance.json", ...]
    - If models only: ["bart_small_*_performance.json", ...]
    - If neither: ["*_performance.json"]
    """
    patterns = []
    
    if models and regions:
        # Both specified: create specific combinations
        for model in models:
            model_safe = model.replace('/', '_').replace('-', '_')
            for region in regions:
                pattern = f"{model_safe}_{region}_{base_pattern}"
                patterns.append(pattern)
    elif regions:
        # Only regions specified
        for region in regions:
            pattern = f"*_{region}_{base_pattern}"
            patterns.append(pattern)
    elif models:
        # Only models specified
        for model in models:
            model_safe = model.replace('/', '_').replace('-', '_')
            pattern = f"{model_safe}_*_{base_pattern}"
            patterns.append(pattern)
    else:
        # Neither specified: match all
        patterns.append(f"*_{base_pattern}")
    
    return patterns


def load_json_files(patterns, subdirectory="json"):
    import glob
    import json
    """Load JSON files matching patterns from volume"""
    json_dir = VOL_MOUNT_PATH / "monitoring_data" / subdirectory
    loaded_data = {}
    
    if not json_dir.exists():
        print(f"⚠️ Directory not found: {json_dir}")
        return loaded_data
    
    for pattern in patterns:
        file_path_pattern = str(json_dir / pattern)
        matching_files = glob.glob(file_path_pattern)
        
        for file_path in matching_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                filename = Path(file_path).name
                loaded_data[filename] = data
                print(f"✅ Loaded: {filename}")
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
    
    return loaded_data


def load_csv_files(patterns, subdirectory="csv"):
    """Load CSV files matching patterns from volume"""
    import pandas as pd
    import glob
    
    csv_dir = VOL_MOUNT_PATH / "monitoring_data" / subdirectory
    loaded_data = {}
    
    if not csv_dir.exists():
        print(f"⚠️ Directory not found: {csv_dir}")
        return loaded_data
    
    for pattern in patterns:
        file_path_pattern = str(csv_dir / pattern)
        matching_files = glob.glob(file_path_pattern)
        
        for file_path in matching_files:
            try:
                df = pd.read_csv(file_path)
                filename = Path(file_path).name
                loaded_data[filename] = df
                print(f"✅ Loaded CSV: {filename} ({len(df)} rows)")
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
    
    return loaded_data


def joules_to_watts(json_files):
    import re
    metrics_in_joules = ['cpu_energy_joules', 'dram_energy_joules', 'total_energy_joules', 'gpu_energy_joules']
    for _, json_content in json_files.items():
        metadata = json_content.get('metadata', {})
        total_execution_time = metadata.get('total_execution_time', None)
        if total_execution_time is not None:
            if isinstance(total_execution_time, str):
                total_execution_time = re.sub(r'[^\d\.eE]', '', total_execution_time)
            try:
                total_execution_time = float(total_execution_time)
            except Exception:
                total_execution_time = None
        if not total_execution_time or total_execution_time == 0:
            continue

        for metric in metrics_in_joules:
            energy = None
            if 'gpu' in metric:
                energy_metric = json_content.get('energy_metrics', {}).get(metric, None)
                if isinstance(energy_metric, dict):
                    energy = sum(energy_metric.values())
                else:
                    energy = energy_metric
            else:
                energy = json_content.get('energy_metrics', {}).get(metric, None)
                if isinstance(energy, dict):
                    energy = sum(energy.values())
            if energy is not None:
                avg_power = energy / total_execution_time
                json_content['energy_metrics'][f"{metric}_watts"] = avg_power
    return json_files



# ============================================================================
# MAIN FUNCTION 
# (Handle monitoring and csv dataset)
# ============================================================================
@app.function(
    image=train_image, 
    cpu=(2, 6), 
    timeout=3600, 
    volumes={str(VOL_MOUNT_PATH): output}
)
@modal.wsgi_app()
def dashboard():
    import pandas as pd
    import dash
    from dash import Dash, dcc, html, Input, Output, State
    import plotly.graph_objs as go
    from dash.exceptions import PreventUpdate
    import re
    from plotly.colors import qualitative
    import json
    from pathlib import Path

    # Models and regions for dropdowns
    model_options = [
        {'label': 'flan-t5-small', 'value': 'flan_t5_small'},
        {'label': 'bart-small', 'value': 'bart_small'}
    ]
    region_options = [
        {'label': 'GCP South Carolina', 'value': 'us-east1'},
        {'label': 'AWS Oregon', 'value': 'us-west1'},
        {'label': 'AZR IOWA', 'value': 'centralus'}
    ]
    
    baseline_region = 'centralus'
    
    regions_names = {
        'us-east1': 'GCP South Carolina',
        'us-west1': 'GCP Oregon',
        'centralus': 'AZR IOWA'
    }

    categories = {
        "Energy Section": ['power.draw [W]', 'temperature.gpu', 'cpu_power', 'dram_power'],
        "Compute Section": [
            'utilization.gpu [%]', 'clocks.current.graphics [MHz]',
            'clocks.current.sm [MHz]', 'sm_active', 'sm_occupancy',
            'tensor_active', 'fp64_active', 'fp32_active', 'fp16_active'
        ],
        "Memory Section": [
            'utilization.memory [%]', 'temperature.memory',
            'clocks.current.memory [MHz]', 'usage.memory [%]','dram_active',
        ],
        "Communication Section": [
            'pcie.link.gen.current', 'pcie.link.width.current',
            'pcie_tx_bytes', 'pcie_rx_bytes', 'nvlink_tx_bytes', 'nvlink_rx_bytes'
        ],
        "System Section": ['cpu_usage','dram_usage']
    }

    def strip_unit(val):
        if pd.isnull(val):
            return None
        match = re.search(r"[-+]?\d*\.?\d+", str(val))
        return float(match.group()) if match else None

    def clean_units(df: pd.DataFrame, unit_fields: list[str]):
        for field in unit_fields:
            if field in df.columns:
                df[f'{field}_raw'] = df[field]
                df[field] = df[field].apply(strip_unit)
        return df

    def load_csv_for_model_region(model, region):
        csv_dir = Path(str(VOL_MOUNT_PATH)) / "monitoring_data" / "csv"
        file_name = f"{model}_{region}_monitor.csv"
        file_path = csv_dir / file_name
        if not file_path.exists():
            return None
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df['index'] = pd.to_numeric(df['index'], errors='coerce')
        df = df.dropna(subset=['index'])
        df['index'] = df['index'].astype(int)
        unit_fields = [
            'clocks.current.memory [MHz]', 'temperature.memory',
            'clocks.current.sm [MHz]', 'temperature.gpu',
            'power.draw [W]', 'utilization.gpu [%]',
            'clocks.current.graphics [MHz]', 'utilization.memory [%]',
            'sm_active', 'sm_occupancy','tensor_active',
            'fp64_active', 'fp32_active', 'fp16_active',
            'usage.utilization [%]', 'dram_active', 'dram_usage',
            'pcie.link.gen.current', 'pcie.link.width.current',
            'pcie_tx_bytes', 'pcie_rx_bytes', 'nvlink_tx_bytes', 'nvlink_rx_bytes',
            'cpu_power', 'cpu_usage', 'dram_power'
        ]
        df = clean_units(df, unit_fields)
        return df
    
    def load_performance_json(model, region):
        json_dir = Path(str(VOL_MOUNT_PATH)) / "monitoring_data" / "json"
        file_name = f"{model}_{region}_performance.json"
        file_path = json_dir / file_name
        if not file_path.exists():
            return None
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def load_inference_json(model, region):
        json_dir = Path(str(VOL_MOUNT_PATH)) / "monitoring_data" / "json"
        file_name = f"{model}_{region}_inference.json"
        file_path = json_dir / file_name
        if not file_path.exists():
            return None
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def joules_to_watts(perf_data, exec_time):
        """Convert energy metrics from Joules to Watts"""
        if not perf_data or not exec_time or exec_time == 0:
            return {}
        
        metrics_in_joules = ['cpu_energy_joules', 'dram_energy_joules', 'total_energy_joules', 'gpu_energy_joules']
        energy_metrics = perf_data.get('energy_metrics', {})
        watts_metrics = {}
        
        for metric in metrics_in_joules:
            energy = energy_metrics.get(metric, None)
            if energy is not None:
                if isinstance(energy, dict):
                    energy = sum(energy.values())
                watts_metrics[f"{metric}_watts"] = energy / exec_time
        
        return watts_metrics
    
    def calculate_regional_difference(value, baseline_value):
        if baseline_value == 0:
            return 0.0
        return ((value - baseline_value) / baseline_value) * 100

    app = Dash(__name__)
    app.layout = html.Div([
        html.H2("GPU & SYSTEM METRICS VISUALIZATION", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.Label("Model:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='model-select',
                    options=model_options,
                    value=model_options[0]['value'],
                    clearable=False
                )
            ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([
                html.Label("Region:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='region-select',
                    options=region_options,
                    value=region_options[0]['value'],
                    clearable=False
                )
            ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([
                html.Label("X-axis:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='x-axis',
                    options=[{'label': 'Timestamp', 'value': 'timestamp'}],
                    value='timestamp',
                    clearable=False
                )
            ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'}),
        ], style={'display': 'flex', 'justifyContent': 'center'}),
        html.Div([
            *[
                html.Div([
                    html.Label(cat, style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id={'type': 'cat-select', 'index': cat},
                        options=[{'label': m, 'value': m} for m in metrics],
                        multi=True,
                        placeholder='To select...'
                    )
                ], style={
                    'width': '19%', 'margin': '10px', 'padding': '10px',
                    'backgroundColor': '#f7f7f7',
                    'borderRadius': '8px'
                }) for cat, metrics in categories.items()
            ]
        ], style={
            'display': 'flex',
            'justifyContent': 'space-between',
            'gap': '20px',
            'padding': '15px',
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
            'borderRadius': '10px',
            'backgroundColor': '#f7f7f7',
            'margin': '0 auto',
            'width': '95%'
        }),
        dcc.Graph(id='indicator-graph', style={'height': '70vh', 'marginTop': '20px'}),
        
        # KPI Cards Section
        html.Div(id='kpi-cards', style={'marginTop': '40px', 'width': '95%', 'margin': '40px auto'})
    ], style={'fontFamily': 'Arial, sans-serif', 'margin': '0 auto', 'width': '95%'})

    @app.callback(
        Output('indicator-graph', 'figure'),
        Input('model-select', 'value'),
        Input('region-select', 'value'),
        Input('x-axis', 'value'),
        Input({'type': 'cat-select', 'index': dash.ALL}, 'value')
    )
    def update_graph(model, region, x_axis, selections):
        df = load_csv_for_model_region(model, region)
        if df is None:
            return go.Figure(layout={'title': f"No data for {model} in {region}"})
        selected_gpus = sorted(df['index'].unique())
        metrics_selected = [m for sel in selections for m in (sel or [])]
        if not selected_gpus or not metrics_selected:
            raise PreventUpdate

        palette = qualitative.Plotly
        color_map = { metric: palette[i % len(palette)]
                    for i, metric in enumerate(metrics_selected) }
        symbol_list = ['circle', 'square', 'diamond', 'cross', 'x',
                    'triangle-up', 'triangle-down', 'pentagon', 'hexagon']
        symbol_map = { gpu: symbol_list[i % len(symbol_list)]
                    for i, gpu in enumerate(selected_gpus) }
        dash_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']

        fig = go.Figure()
        for gpu in selected_gpus:
            dfg = df[df['index'] == gpu]
            for metric in metrics_selected:
                if metric in dfg.columns:
                    raw_col = f"{metric}_raw" if f"{metric}_raw" in dfg.columns else None
                    hover_text = [
                        f"Timestamp: {t.strftime('%H:%M:%S')}.{int(t.microsecond / 100000)}<br>{metric}: {dfg[raw_col].iloc[i] if raw_col else val}"
                        for i, (t, val) in enumerate(zip(dfg[x_axis], dfg[metric]))
                    ]
                    fig.add_trace(go.Scatter(
                        x=dfg[x_axis],
                        y=dfg[metric],
                        mode='lines+markers',
                        name=f"GPU{gpu} — {metric}",
                        legendgroup=f"GPU{gpu}",
                        line=dict(
                            color=color_map[metric],
                            dash=dash_styles[gpu % len(dash_styles)],
                            shape='linear'
                        ),
                        marker=dict(
                            symbol=symbol_map[gpu],
                            size=6,
                            color=color_map[metric]
                        ),
                        hoverinfo='text',
                        hovertext=hover_text
                    ))

        fig.update_layout(
            title=f"Metrics Trend for {model} in {region}",
            xaxis=dict(
                title="Time",
                tickformat="%H:%M:%S",
                tickangle=30,
                showgrid=False,
                gridcolor='#e0e0e0'
            ),
            yaxis=dict( 
                title="Value",
                showgrid=False,
                gridcolor='#e0e0e0'
            ),
            legend_title="Legend",
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    
    @app.callback(
        Output('kpi-cards', 'children'),
        Input('model-select', 'value'),
        Input('region-select', 'value')
    )
    def update_kpis(model, region):
        # Load performance and inference data
        perf_data = load_performance_json(model, region)
        inf_data = load_inference_json(model, region)
        baseline_perf = load_performance_json(model, baseline_region)
        baseline_inf = load_inference_json(model, baseline_region)
        
        if not perf_data:
            return html.Div("No performance data available", style={'textAlign': 'center', 'padding': '20px'})
        
        # Extract execution time
        exec_time_str = perf_data.get('metadata', {}).get('total_execution_time', '0')
        exec_time = float(re.sub(r'[^\d\.eE]', '', str(exec_time_str))) if exec_time_str else 0
        
        # Convert Joules to Watts
        watts_metrics = joules_to_watts(perf_data, exec_time)
        energy_metrics = perf_data.get('energy_metrics', {})
        
        # Get baseline values
        baseline_exec_time = 0
        baseline_watts = {}
        baseline_energy = {}
        if baseline_perf and region != baseline_region:
            baseline_exec_str = baseline_perf.get('metadata', {}).get('total_execution_time', '0')
            baseline_exec_time = float(re.sub(r'[^\d\.eE]', '', str(baseline_exec_str))) if baseline_exec_str else 0
            baseline_watts = joules_to_watts(baseline_perf, baseline_exec_time)
            baseline_energy = baseline_perf.get('energy_metrics', {})
        
        # Calculate performance KPIs
        total_energy = watts_metrics.get('total_energy_joules_watts', 0)
        carbon_emissions = energy_metrics.get('carbon_emissions_kg', 0)
        
        # Calculate deltas
        energy_delta = None
        carbon_delta = None
        exec_delta = None
        
        if region != baseline_region and baseline_perf:
            baseline_total = baseline_watts.get('total_energy_joules_watts', 0)
            baseline_carbon = baseline_energy.get('carbon_emissions_kg', 0)
            
            if baseline_total > 0:
                energy_delta = calculate_regional_difference(total_energy, baseline_total)
            if baseline_carbon > 0:
                carbon_delta = calculate_regional_difference(carbon_emissions, baseline_carbon)
            if baseline_exec_time > 0:
                exec_delta = calculate_regional_difference(exec_time, baseline_exec_time)
        
        # Calculate inference KPIs
        inf_kpis = []
        if inf_data:
            num_samples = inf_data.get('num_samples', 0)
            throughput = inf_data.get('throughput_samples_per_second', 0)
            comp = inf_data.get('compression_metrics', {})
            avg_in = comp.get('avg_input_tokens', 0)
            avg_out = comp.get('avg_output_tokens', 0)
            total_tokens = num_samples * (avg_in + avg_out)
            
            total_energy_joules = energy_metrics.get('total_energy_joules', 0)
            if isinstance(total_energy_joules, dict):
                total_energy_joules = sum(total_energy_joules.values())
            
            carbon_per_token = carbon_emissions / total_tokens if total_tokens else 0
            energy_per_token = total_energy_joules / total_tokens if total_tokens else 0
            carbon_per_sample = carbon_emissions / num_samples if num_samples else 0
            carbon_throughput_ratio = throughput / carbon_emissions if carbon_emissions else 0
            
            # Calculate baseline inference metrics
            cpt_delta = None
            ept_delta = None
            cps_delta = None
            ctr_delta = None
            
            if baseline_inf and baseline_perf and region != baseline_region:
                b_num_samples = baseline_inf.get('num_samples', 0)
                b_throughput = baseline_inf.get('throughput_samples_per_second', 0)
                b_comp = baseline_inf.get('compression_metrics', {})
                b_avg_in = b_comp.get('avg_input_tokens', 0)
                b_avg_out = b_comp.get('avg_output_tokens', 0)
                b_total_tokens = b_num_samples * (b_avg_in + b_avg_out)
                
                b_carbon = baseline_energy.get('carbon_emissions_kg', 0)
                b_total_energy = baseline_energy.get('total_energy_joules', 0)
                if isinstance(b_total_energy, dict):
                    b_total_energy = sum(b_total_energy.values())
                
                b_carbon_per_token = b_carbon / b_total_tokens if b_total_tokens else 0
                b_energy_per_token = b_total_energy / b_total_tokens if b_total_tokens else 0
                b_carbon_per_sample = b_carbon / b_num_samples if b_num_samples else 0
                b_carbon_throughput_ratio = b_throughput / b_carbon if b_carbon else 0
                
                if b_carbon_per_token > 0:
                    cpt_delta = calculate_regional_difference(carbon_per_token, b_carbon_per_token)
                if b_energy_per_token > 0:
                    ept_delta = calculate_regional_difference(energy_per_token, b_energy_per_token)
                if b_carbon_per_sample > 0:
                    cps_delta = calculate_regional_difference(carbon_per_sample, b_carbon_per_sample)
                if b_carbon_throughput_ratio > 0:
                    ctr_delta = calculate_regional_difference(carbon_throughput_ratio, b_carbon_throughput_ratio)
            
            inf_kpis = [
                ('Carbon / Token', f"{carbon_per_token:.6f} kg", cpt_delta, True),
                ('Energy / Token', f"{energy_per_token:.2f} J", ept_delta, True),
                ('Carbon / Sample', f"{carbon_per_sample:.6f} kg", cps_delta, True),
                ('Throughput / CO₂', f"{carbon_throughput_ratio:.3f} samp·s⁻¹·kg⁻¹", ctr_delta, False)
            ]
        
        # Helper function to create delta badge
        def delta_badge(delta, good_when_lower=False):
            if delta is None:
                return ""
            is_good = (delta < 0) if good_when_lower else (delta > 0)
            color = '#10b981' if is_good else '#ef4444'
            arrow = "↓" if delta < 0 else "↑"
            return html.Span(f" {arrow} {abs(delta):.1f}%", style={'color': color, 'fontSize': '12px', 'fontWeight': 'bold'})
        
        # Create KPI card layout
        kpi_section = html.Div([
            html.H3("Performance Metrics", style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            # Main Performance KPIs
            html.Div([
                html.Div([
                    html.Div([
                        html.P("Total Energy", style={'fontSize': '12px', 'color': '#666', 'margin': '0'}),
                        html.Div([
                            html.H3(f"{total_energy:,.2f} W", style={'fontSize': '24px', 'margin': '5px 0', 'color': '#3b82f6'}),
                            delta_badge(energy_delta, True)
                        ], style={'display': 'flex', 'alignItems': 'baseline', 'gap': '8px'})
                    ])
                ], style={
                    'flex': '1',
                    'padding': '20px',
                    'backgroundColor': '#f7f7f7',
                    'borderRadius': '8px',
                    'border': '1px solid #e0e0e0'
                }),
                
                html.Div([
                    html.Div([
                        html.P("Carbon Emissions", style={'fontSize': '12px', 'color': '#666', 'margin': '0'}),
                        html.Div([
                            html.H3(f"{carbon_emissions:.4f} kg", style={'fontSize': '24px', 'margin': '5px 0', 'color': '#10b981'}),
                            delta_badge(carbon_delta, True)
                        ], style={'display': 'flex', 'alignItems': 'baseline', 'gap': '8px'})
                    ])
                ], style={
                    'flex': '1',
                    'padding': '20px',
                    'backgroundColor': '#f7f7f7',
                    'borderRadius': '8px',
                    'border': '1px solid #e0e0e0'
                }),
                
                html.Div([
                    html.Div([
                        html.P("Execution Time", style={'fontSize': '12px', 'color': '#666', 'margin': '0'}),
                        html.Div([
                            html.H3(f"{exec_time:.2f} s", style={'fontSize': '24px', 'margin': '5px 0', 'color': '#f59e0b'}),
                            delta_badge(exec_delta, True)
                        ], style={'display': 'flex', 'alignItems': 'baseline', 'gap': '8px'})
                    ])
                ], style={
                    'flex': '1',
                    'padding': '20px',
                    'backgroundColor': '#f7f7f7',
                    'borderRadius': '8px',
                    'border': '1px solid #e0e0e0'
                })
            ], style={
                'display': 'flex',
                'gap': '20px',
                'marginBottom': '30px'
            }),
            
            # Inference Efficiency KPIs (if available)
            html.Div([
                html.H3("Inference Efficiency", style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.Div([
                    html.Div([
                        html.P(label, style={'fontSize': '12px', 'color': '#666', 'margin': '0'}),
                        html.Div([
                            html.H4(value, style={'fontSize': '16px', 'margin': '5px 0'}),
                            delta_badge(delta, good_when_lower)
                        ], style={'display': 'flex', 'alignItems': 'baseline', 'gap': '8px'})
                    ], style={
                        'padding': '15px',
                        'backgroundColor': '#f7f7f7',
                        'borderRadius': '8px',
                        'border': '1px solid #e0e0e0'
                    })
                    for label, value, delta, good_when_lower in inf_kpis
                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
                    'gap': '15px'
                })
            ]) if inf_kpis else html.Div()
        ])
        
        return kpi_section

    return app.server



# To run: modal run unsloth_finetune.py::main
@app.local_entrypoint()
def main():  # Add parameter to control clearing
    import os
    import re
    
    """
    Reference: https://modal.com/docs/guide/region-selection#united-states-us
    Reference: https://gist.github.com/atyachin/a011edf76df66c5aa1eac0cdca412ea9
    
    Broad            Specific             Description           Available (ElectricityMaps)
    ==========================================================================================
    "us-east"           "us-east-1"          AWS Virginia           ✅
                        "us-east-2"          AWS Ohio               ✅
                        "us-east1"           GCP South Carolina     ✅
                        "us-east4"           GCP Virginia           ✅
                        "us-east5"           GCP Ohio
                        "us-ashburn-1"       OCI Virginia
                        "eastus"             AZR Virginia           ✅
                        "eastus2"            AZR Virginia           ✅
    ------------------------------------------------------------------------------------------
    "us-central"        "us-central1"        GCP Iowa               ✅
                        "us-chicago-1"       OCI Chicago        
                        "us-phoenix-1"       OCI Phoenix
                        "centralus"          AZR Iowa               ✅
                        "northcentralus"     AZR Illinois           ✅
                        "southcentralus"     AZR Texas              ✅
                        "westcentralus"      AZR Wyoming            ✅
    ------------------------------------------------------------------------------------------
    "us-west"           "us-west-1"          AWS California         ✅
                        "us-west-2"          AWS Oregon             ✅
                        "us-west1"           GCP Oregon             ✅
                        "us-west3"           GCP Utah
                        "us-west4"           GCP Nevada
                        "us-sanjose-1"       OCI San Jose
                        "westus"             AZR California         ✅
                        "westus2"            AZR Washington         ✅
                        "westus3"            AZR Phoenix            ✅
    ==========================================================================================
    """
               
    # Print Modal Environement
    print(f"\n{'='*70}")
    print(f"Running Main Function...")
    print(f"{'='*70}")
    
    # Optional: Reset the restart tracker at the beginning
    # reset_restart_tracker.remote()
    
    # Create Output directory locally if it doesn't exist
    local_output_dir = "Output"
    os.makedirs(local_output_dir, exist_ok=True)
    print(f"Output directory created: {local_output_dir}")
    
    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    
    models = [model['name'] for model in SMALL_MODELS.values()]
    print(f"  Models: {models}")
    print(f"  Regions (Test): {TEST_REGIONS}")
    print(f"{'='*70}\n")
    
    # Obtain broad region
    for broad_region in TEST_REGIONS:
        
        print(f"\n{'='*70}")
        print(f"🌍 Processing Broad Region: {broad_region}")
        print(f"{'='*70}\n")
        
        # Run monitoring and training for both models
        # Call .with_options() on the CLASS, then instantiate
        RegionalGPUClass = GPUMonitoringClass.with_options(region=AVAILABLE_REGIONS[broad_region])
        gpu_instance = RegionalGPUClass()
        result = gpu_instance.monitor_workload.remote(models_to_train = models)
    
        # Handle case where method returns None
        if result is None:
            csv_files, json_files = {}, {}
        else:
            csv_files, json_files = result
        
        # Save CSV Files Locally
        if csv_files and len(csv_files) > 0:  
            print(f"Training completed. CSV files generated: {list(csv_files.keys())}")
            for csv_file_name, csv_content in csv_files.items():              
                if csv_content:
                    # Save to local Output directory
                    local_file_path = os.path.join(local_output_dir, csv_file_name)
                    with open(local_file_path, 'w', encoding='utf-8') as f:
                        f.write(csv_content)
                    print(f"✅ CSV saved locally: {local_file_path}")
                else:
                    print(f"❌ No CSV content received for {csv_file_name}")
        else:
            print("❌ No CSV files generated")
        
        # Save JSON Performance Files Locally
        if json_files and len(json_files) > 0:
            print(f"Performance JSON files generated: {list(json_files.keys())}")
            for json_file_name, json_content in json_files.items():
                if json_content:
                    # Save to local Output directory
                    local_json_path = os.path.join(local_output_dir, json_file_name)
                    with open(local_json_path, 'w', encoding='utf-8') as f:
                        f.write(json_content)
                    print(f"✅ JSON saved locally: {local_json_path}")
                else:
                    print(f"❌ No JSON content received for {json_file_name}")
        else:
            print("❌ No JSON performance files generated")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ Training Complete for All Regions")
    print(f"{'='*70}")

    # List files in the local Output directory
    if os.path.exists(local_output_dir):
        local_csv_files = [f for f in os.listdir(local_output_dir) if f.endswith('.csv')]
        local_json_files = [f for f in os.listdir(local_output_dir) if f.endswith('.json')]
        print(f"CSV files saved locally in {local_output_dir}: {local_csv_files}")
        print(f"JSON files saved locally in {local_output_dir}: {local_json_files}")
    else:
        print("No local output directory found")

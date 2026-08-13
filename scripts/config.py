"""Run-time constants for the command-line entrypoints."""

from pathlib import Path

from src.sae.base import TrainingConfig

# DIRECTORIES
DEFAULT_EXTRACT_DIR = Path("data/raw")
DEFAULT_SYNTHETIC_DIR = Path("data/synthetic")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_PROCESSED_SYNTHETIC_DIR = Path("data/processed_synthetic")
DEFAULT_BMC_RESULTS_DIR = Path("data/bmc_results")
DEFAULT_BMC_SYNTHETIC_RESULTS_DIR = Path("data/bmc_synthetic_results")
DEFAULT_SAE_RESULTS_DIR = Path("data/sae_results")
DEFAULT_SAE_SYNTHETIC_RESULTS_DIR = Path("data/sae_synthetic_results")
DEFAULT_FIGURES_DIR = Path("figures")

# SYNTHETIC DATA
DEFAULT_N_SAMPLES = 200
DEFAULT_DIM = 20
DEFAULT_SEED = 7
DEFAULT_SIGMA2 = 0.3

# LLM ACTIVATIONS
DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_CONCEPT = "days"
DEFAULT_LAYER = 16
DEFAULT_BATCH_SIZE = 4
DEFAULT_DTYPE = "float16"
DEFAULT_PCA_COMPONENTS = 20

# AUTOENCODERS
DEFAULT_NUM_FEATURES = 10
TRAINING_CONFIG = TrainingConfig(
    num_epochs=5000,
    batch_size=64,
    lr=3e-3,
    l1_coeff=0.15,
    normalise_every=0,
)
DEFAULT_ALIVE_THRESHOLD = 0.1
DEFAULT_MAX_CONCURRENT_MODELS = 8

CONCEPTS = {
    # --- periodic / circular (expected to be non-linear) ---
    "days": {
        "labels": [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        "templates": [
            "Today is {day}.",
            "The meeting is scheduled for {day}.",
            "She goes to the gym every {day}.",
            "The deadline falls on {day}.",
            "{day} is my favourite day of the week.",
            "We always have dinner together on {day}.",
            "The report is due on {day} morning.",
            "He called me last {day}.",
            "The store is closed on {day}.",
            "My birthday this year falls on a {day}.",
            "Classes resume on {day}.",
            "The event was postponed to {day}.",
            "She woke up early on {day} for the trip.",
            "Every {day} the market opens at dawn.",
            "The team celebrates wins on {day} evening.",
        ],
    },
    # --- periodic / circular (expected to be non-linear) ---
    "months": {
        "labels": [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        "templates": [
            "The event happens in {month}.",
            "She was born in {month}.",
            "The fiscal year starts in {month}.",
            "We always go on holiday in {month}.",
            "Spring arrives in {month}.",
            "The company was founded in {month}.",
            "Temperatures peak in {month}.",
            "The semester begins in {month}.",
            "He submitted the paper in {month}.",
            "The festival takes place in {month}.",
            "Sales usually spike in {month}.",
            "She graduated in {month}.",
            "The project deadline is in {month}.",
            "The new product launches in {month}.",
            "Construction starts in {month}.",
        ],
    },
    # --- linear control set (expected to be linear) ---
    "sentiment": {
        "labels": [
            "terrible",
            "bad",
            "mediocre",
            "okay",
            "good",
            "great",
            "excellent",
        ],
        "templates": [
            "The film was {word}.",
            "I thought the meal was {word}.",
            "Her performance was {word}.",
            "The customer service was {word}.",
            "He considered the result {word}.",
            "The product quality is {word}.",
            "The weather today is {word}.",
            "Overall the experience was {word}.",
            "His attitude has been {word} lately.",
            "The presentation was {word}.",
        ],
    },
    # --- helical (circular + linear components from modular arithmetic) ---
    "numbers": {
        "labels": [str(i) for i in range(10)],
        "templates": [
            "The number is {number}.",
            "She has {number} cats.",
            "The code is {number}.",
            "I will arrive in {number} minutes.",
            "The answer to the problem is {number}.",
            "The recipe calls for {number} eggs.",
            "He scored {number} goals.",
            "The package weighs {number} kilograms.",
            "The event starts at {number} o'clock.",
            "The population is {number} million.",
            "The temperature is {number} degrees.",
            "The book has {number} pages.",
            "The team won by {number} points.",
            "She is {number} years old.",
            "The flight departs in {number} hours.",
        ],
    },
    # --- linear control set (ordinal scale, no periodicity) ---
    "size": {
        "labels": [
            "tiny",
            "small",
            "medium",
            "large",
            "huge",
            "enormous",
        ],
        "templates": [
            "The house was {size}.",
            "She ordered a {size} coffee.",
            "The difference in quality was {size}.",
            "They found a {size} rock on the beach.",
            "The company experienced a {size} loss.",
            "He caught a {size} fish in the lake.",
            "The crowd at the concert was {size}.",
            "The gap between the two teams was {size}.",
            "Her collection of stamps was {size}.",
            "The impact of the decision was {size}.",
            "The dog they adopted was {size}.",
            "The portion of food was {size}.",
            "The mountain in the distance looked {size}.",
            "The budget for the project was {size}.",
            "The improvement over last year was {size}.",
        ],
    },
}

SHORT_LABELS = {
    "monday": "Mon",
    "tuesday": "Tue",
    "wednesday": "Wed",
    "thursday": "Thu",
    "friday": "Fri",
    "saturday": "Sat",
    "sunday": "Sun",
    "january": "Jan",
    "february": "Feb",
    "march": "Mar",
    "april": "Apr",
    "may": "May",
    "june": "Jun",
    "july": "Jul",
    "august": "Aug",
    "september": "Sep",
    "october": "Oct",
    "november": "Nov",
    "december": "Dec",
    "terrible": "terr.",
    "mediocre": "med.",
    "excellent": "exc.",
}

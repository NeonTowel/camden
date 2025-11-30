//! Image tagging using configurable ONNX models (MobileNetV2, EfficientNet, ViT, etc.).
//!
//! Supports multiple ImageNet-based models with configurable input sizes and normalization.

use super::config::ModelInputSpec;
use super::runtime::{load_session, preprocess_image_with_layout, softmax, ClassifierError};
use ort::session::Session;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Minimum confidence threshold for tags (0.0-1.0).
/// Tags below this threshold are filtered out.
/// 0.6 = 60% confidence
pub const MIN_TAG_CONFIDENCE: f32 = 0.6;

/// Configuration for the tagging classifier.
#[derive(Clone, Debug)]
pub struct TaggingConfig {
    /// Input image width
    pub input_width: u32,
    /// Input image height
    pub input_height: u32,
    /// Whether to apply ImageNet normalization
    pub normalize: bool,
    /// Input tensor layout ("NCHW" or "NHWC")
    pub layout: String,
}

impl Default for TaggingConfig {
    fn default() -> Self {
        Self {
            input_width: 224,
            input_height: 224,
            normalize: true,
            layout: "NCHW".to_string(),
        }
    }
}

impl TaggingConfig {
    /// Create config from model input spec.
    pub fn from_specs(input: &ModelInputSpec) -> Self {
        Self {
            input_width: input.width,
            input_height: input.height,
            normalize: input.normalize,
            layout: input.layout.clone(),
        }
    }
}

/// Image tagging classifier using ImageNet models.
pub struct TaggingClassifier {
    session: Session,
    config: TaggingConfig,
}

impl TaggingClassifier {
    /// Load the tagging classifier from an ONNX model file with default config.
    pub fn new(model_path: &Path) -> Result<Self, ClassifierError> {
        Self::with_config(model_path, TaggingConfig::default())
    }

    /// Load the tagging classifier with custom configuration.
    pub fn with_config(model_path: &Path, config: TaggingConfig) -> Result<Self, ClassifierError> {
        let session = load_session(model_path)?;
        Ok(Self { session, config })
    }

    /// Classify an image and return the top tags.
    pub fn classify(
        &mut self,
        image_path: &Path,
        max_tags: usize,
    ) -> Result<Vec<ImageTag>, ClassifierError> {
        let input_size = (self.config.input_width as i32, self.config.input_height as i32);
        
        // Preprocess with model-specific settings including layout
        let input = preprocess_image_with_layout(
            image_path, 
            input_size, 
            self.config.normalize,
            &self.config.layout
        )?;

        // Get input name from model
        let input_name = self
            .session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "input".to_string());

        // Create tensor from ndarray
        let input_tensor =
            ort::value::Tensor::from_array(input).map_err(ClassifierError::Ort)?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![input_name => input_tensor])
            .map_err(ClassifierError::Ort)?;

        // Extract output tensor
        let output = outputs
            .values()
            .next()
            .ok_or_else(|| ClassifierError::Processing("no output tensor found".into()))?;

        // Extract as tuple (shape, data slice)
        let (shape, logits_slice) = output
            .try_extract_tensor::<f32>()
            .map_err(ClassifierError::Ort)?;

        // Flatten to 1D if needed (some models output [batch_size, num_classes])
        let logits: Vec<f32> = if shape.len() > 1 && shape[0] == 1 {
            // Output is [1, num_classes], extract just the class scores
            logits_slice.to_vec()
        } else if shape.len() > 1 {
            // Unexpected shape, just use as is
            logits_slice.to_vec()
        } else {
            logits_slice.to_vec()
        };
        
        // Check if logits already sum to ~1.0 (indicating they're probabilities, not logits)
        let logits_sum: f32 = logits.iter().sum();
        let is_already_probabilities = (logits_sum - 1.0).abs() < 0.01; // Allow small tolerance
        
        // Apply softmax only if not already probabilities
        let probabilities = if is_already_probabilities {
            logits.clone()
        } else {
            softmax(&logits)
        };

        // Get top-k predictions
        let mut indexed: Vec<(usize, f32)> = probabilities.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let tags: Vec<ImageTag> = indexed
            .into_iter()
            .take(max_tags)
            .filter(|(_, score)| *score >= MIN_TAG_CONFIDENCE) // Filter by confidence threshold
            .filter_map(|(idx, score)| {
                IMAGENET_LABELS.get(idx).map(|label| {
                    let (category, clean_label) = categorize_label(label);
                    ImageTag {
                        name: clean_label.to_lowercase().replace(' ', "-"),
                        label: clean_label.to_string(),
                        confidence: score,
                        category,
                    }
                })
            })
            .collect();

        Ok(tags)
    }
}

/// A generated image tag with metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageTag {
    /// Normalized tag name (lowercase, hyphenated)
    pub name: String,
    /// Human-readable label
    pub label: String,
    /// Model confidence (0.0-1.0)
    pub confidence: f32,
    /// Tag category
    pub category: TagCategory,
}

/// Category for organizing tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TagCategory {
    /// Animals
    Animal,
    /// Vehicles and transportation
    Vehicle,
    /// Food and drinks
    Food,
    /// Nature and landscapes
    Nature,
    /// People and body parts
    Person,
    /// Buildings and structures
    Structure,
    /// Electronics and devices
    Device,
    /// Furniture and household items
    Furniture,
    /// Sports and activities
    Sport,
    /// Clothing and accessories
    Clothing,
    /// Musical instruments
    Music,
    /// General objects
    Object,
}

impl std::fmt::Display for TagCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Animal => write!(f, "Animal"),
            Self::Vehicle => write!(f, "Vehicle"),
            Self::Food => write!(f, "Food"),
            Self::Nature => write!(f, "Nature"),
            Self::Person => write!(f, "Person"),
            Self::Structure => write!(f, "Structure"),
            Self::Device => write!(f, "Device"),
            Self::Furniture => write!(f, "Furniture"),
            Self::Sport => write!(f, "Sport"),
            Self::Clothing => write!(f, "Clothing"),
            Self::Music => write!(f, "Music"),
            Self::Object => write!(f, "Object"),
        }
    }
}

/// Categorize a label and clean it up.
fn categorize_label(label: &str) -> (TagCategory, &str) {
    // Extract the primary term (before comma if present)
    let primary = label.split(',').next().unwrap_or(label).trim();

    // Simple keyword-based categorization
    let lower = label.to_lowercase();

    let category = if ANIMAL_KEYWORDS.iter().any(|k| lower.contains(k)) {
        TagCategory::Animal
    } else if VEHICLE_KEYWORDS.iter().any(|k| lower.contains(k)) {
        TagCategory::Vehicle
    } else if FOOD_KEYWORDS.iter().any(|k| lower.contains(k)) {
        TagCategory::Food
    } else if NATURE_KEYWORDS.iter().any(|k| lower.contains(k)) {
        TagCategory::Nature
    } else if PERSON_KEYWORDS.iter().any(|k| lower.contains(k)) {
        TagCategory::Person
    } else if STRUCTURE_KEYWORDS.iter().any(|k| lower.contains(k)) {
        TagCategory::Structure
    } else if DEVICE_KEYWORDS.iter().any(|k| lower.contains(k)) {
        TagCategory::Device
    } else if FURNITURE_KEYWORDS.iter().any(|k| lower.contains(k)) {
        TagCategory::Furniture
    } else if SPORT_KEYWORDS.iter().any(|k| lower.contains(k)) {
        TagCategory::Sport
    } else if CLOTHING_KEYWORDS.iter().any(|k| lower.contains(k)) {
        TagCategory::Clothing
    } else if MUSIC_KEYWORDS.iter().any(|k| lower.contains(k)) {
        TagCategory::Music
    } else {
        TagCategory::Object
    };

    (category, primary)
}

// Category keyword lists
const ANIMAL_KEYWORDS: &[&str] = &[
    "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "pig", "chicken", "duck", "goose",
    "rabbit", "mouse", "rat", "hamster", "guinea", "snake", "lizard", "turtle", "frog", "toad",
    "spider", "insect", "butterfly", "bee", "ant", "beetle", "lion", "tiger", "bear", "wolf",
    "fox", "deer", "elephant", "giraffe", "zebra", "monkey", "ape", "gorilla", "whale", "dolphin",
    "shark", "seal", "penguin", "owl", "eagle", "hawk", "parrot", "terrier", "retriever",
    "shepherd", "bulldog", "poodle", "beagle", "hound", "spaniel", "collie", "husky", "malamute",
    "corgi", "dachshund", "chihuahua", "pug", "boxer", "mastiff", "rottweiler", "doberman",
    "persian", "siamese", "tabby", "kitten", "puppy",
];

const VEHICLE_KEYWORDS: &[&str] = &[
    "car", "truck", "bus", "train", "plane", "airplane", "aircraft", "helicopter", "boat", "ship",
    "bicycle", "motorcycle", "scooter", "van", "taxi", "ambulance", "fire engine", "police",
    "tractor", "forklift", "crane", "bulldozer", "limousine", "convertible", "sedan", "suv",
    "minivan", "wagon", "jeep", "pickup", "trailer", "carriage", "locomotive", "subway", "tram",
    "ferry", "yacht", "canoe", "kayak", "speedboat", "sailboat",
];

const FOOD_KEYWORDS: &[&str] = &[
    "food", "fruit", "vegetable", "meat", "bread", "cake", "pizza", "burger", "sandwich", "salad",
    "soup", "pasta", "rice", "noodle", "sushi", "egg", "cheese", "milk", "butter", "cream", "ice",
    "chocolate", "candy", "cookie", "pie", "donut", "muffin", "croissant", "bagel", "pretzel",
    "apple", "orange", "banana", "grape", "strawberry", "cherry", "peach", "pear", "watermelon",
    "pineapple", "mango", "lemon", "lime", "coconut", "avocado", "tomato", "potato", "carrot",
    "broccoli", "lettuce", "onion", "pepper", "mushroom", "corn", "bean", "pea", "cucumber",
    "eggplant", "zucchini", "squash", "pumpkin", "cabbage", "spinach", "celery", "garlic",
];

const NATURE_KEYWORDS: &[&str] = &[
    "tree", "flower", "plant", "grass", "forest", "mountain", "hill", "valley", "river", "lake",
    "ocean", "sea", "beach", "island", "desert", "jungle", "swamp", "meadow", "field", "garden",
    "park", "cliff", "waterfall", "volcano", "glacier", "canyon", "cave", "rock", "stone",
    "sand", "snow", "ice", "cloud", "sky", "sun", "moon", "star", "rainbow", "sunset", "sunrise",
    "landscape", "seascape", "coast", "shore", "reef", "coral",
];

const PERSON_KEYWORDS: &[&str] = &[
    "person", "man", "woman", "child", "baby", "boy", "girl", "face", "head", "hand", "foot",
    "arm", "leg", "body", "portrait", "crowd", "group", "family", "couple", "wedding", "groom",
    "bride", "athlete", "player", "dancer", "singer", "actor", "model",
];

const STRUCTURE_KEYWORDS: &[&str] = &[
    "building", "house", "apartment", "skyscraper", "tower", "bridge", "tunnel", "road", "street",
    "highway", "path", "sidewalk", "fence", "wall", "gate", "door", "window", "roof", "chimney",
    "church", "temple", "mosque", "cathedral", "castle", "palace", "museum", "library", "school",
    "hospital", "hotel", "restaurant", "shop", "store", "mall", "market", "stadium", "arena",
    "theater", "cinema", "airport", "station", "port", "harbor", "dam", "lighthouse", "monument",
    "statue", "fountain", "arch", "dome", "pyramid", "pagoda",
];

const DEVICE_KEYWORDS: &[&str] = &[
    "computer", "laptop", "desktop", "monitor", "screen", "keyboard", "mouse", "phone",
    "smartphone", "tablet", "camera", "television", "tv", "radio", "speaker", "headphone",
    "microphone", "printer", "scanner", "projector", "remote", "controller", "console", "game",
    "watch", "clock", "calculator", "battery", "charger", "cable", "wire", "plug", "socket",
    "switch", "button", "dial", "meter", "gauge", "sensor", "antenna", "satellite", "robot",
    "drone", "ipod", "cd", "dvd", "usb", "disk", "drive", "modem", "router", "server",
];

const FURNITURE_KEYWORDS: &[&str] = &[
    "chair", "table", "desk", "sofa", "couch", "bed", "mattress", "pillow", "blanket", "sheet",
    "curtain", "carpet", "rug", "lamp", "chandelier", "shelf", "bookcase", "cabinet", "drawer",
    "wardrobe", "closet", "mirror", "frame", "vase", "pot", "plant", "basket", "box", "container",
    "bin", "trash", "bucket", "broom", "mop", "vacuum", "iron", "fan", "heater", "air",
    "refrigerator", "freezer", "oven", "stove", "microwave", "toaster", "blender", "mixer",
    "dishwasher", "washer", "dryer", "sink", "faucet", "toilet", "bathtub", "shower",
];

const SPORT_KEYWORDS: &[&str] = &[
    "ball", "bat", "racket", "net", "goal", "basket", "hoop", "pool", "tennis", "golf",
    "football", "soccer", "basketball", "baseball", "volleyball", "hockey", "cricket", "rugby",
    "boxing", "wrestling", "martial", "karate", "judo", "fencing", "archery", "shooting",
    "cycling", "swimming", "diving", "surfing", "skiing", "snowboard", "skating", "running",
    "jogging", "walking", "hiking", "climbing", "gym", "fitness", "yoga", "weight", "dumbbell",
    "barbell", "treadmill", "bicycle", "helmet", "glove", "paddle", "oar", "ski", "pole",
];

const CLOTHING_KEYWORDS: &[&str] = &[
    "shirt", "pants", "jeans", "shorts", "skirt", "dress", "suit", "jacket", "coat", "sweater",
    "hoodie", "vest", "blouse", "top", "bottom", "underwear", "sock", "shoe", "boot", "sandal",
    "slipper", "sneaker", "heel", "flat", "hat", "cap", "beanie", "scarf", "glove", "mitten",
    "belt", "tie", "bow", "necklace", "bracelet", "ring", "earring", "watch", "glasses",
    "sunglasses", "bag", "purse", "backpack", "wallet", "umbrella", "mask", "uniform", "costume",
    "bikini", "swimsuit", "robe", "pajama", "apron", "jersey", "kimono", "sari",
];

const MUSIC_KEYWORDS: &[&str] = &[
    "guitar", "piano", "keyboard", "drum", "violin", "cello", "bass", "flute", "clarinet",
    "saxophone", "trumpet", "trombone", "horn", "harmonica", "accordion", "banjo", "ukulele",
    "harp", "organ", "synthesizer", "microphone", "speaker", "amplifier", "mixer", "turntable",
    "record", "vinyl", "cd", "cassette", "headphone", "earphone", "baton", "conductor", "orchestra",
    "band", "choir", "concert", "stage", "studio", "instrument", "musical",
];

// ImageNet 1000 class labels (abbreviated - full list would be 1000 entries)
// This is a subset of common labels; for production, load from a file
const IMAGENET_LABELS: &[&str] = &[
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
    "electric ray", "stingray", "cock", "hen", "ostrich",
    "brambling", "goldfinch", "house finch", "junco", "indigo bunting",
    "American robin", "bulbul", "jay", "magpie", "chickadee",
    "American dipper", "kite", "bald eagle", "vulture", "great grey owl",
    "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl",
    "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle",
    "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana",
    "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
    "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
    "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake",
    "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake",
    "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba",
    "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder", "trilobite",
    "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider",
    "southern black widow", "tarantula", "wolf spider", "tick", "centipede",
    "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peacock",
    "quail", "partridge", "grey parrot", "macaw", "sulphur-crested cockatoo",
    "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird",
    "jacamar", "toucan", "duck", "red-breasted merganser", "goose",
    "black swan", "tusker", "echidna", "platypus", "wallaby",
    "koala", "wombat", "jellyfish", "sea anemone", "brain coral",
    "flatworm", "nematode", "conch", "snail", "slug",
    "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
    "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish",
    "hermit crab", "isopod", "white stork", "black stork", "spoonbill",
    "flamingo", "little blue heron", "great egret", "bittern", "crane",
    "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone",
    "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican",
    "king penguin", "albatross", "grey whale", "killer whale", "dugong",
    "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese",
    "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback",
    "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound",
    "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi",
    "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound",
    "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier",
    "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
    "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier",
    "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier",
    "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier",
    "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso",
    "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever",
    "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter",
    "Brittany", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel",
    "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael",
    "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog",
    "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres", "Rottweiler",
    "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
    "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
    "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute",
    "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug",
    "Leonberger", "Newfoundland", "Pyrenean Mountain Dog", "Samoyed", "Pomeranian",
    "Chow Chow", "Keeshond", "Griffon Bruxellois", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi",
    "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog", "grey wolf",
    "Alaskan tundra wolf", "red wolf", "coyote", "dingo",
    "dhole", "African wild dog", "hyena", "red fox", "kit fox",
    "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat",
    "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard",
    "snow leopard", "jaguar", "lion", "tiger", "cheetah",
    "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
    "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle",
    "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly",
    "bee", "ant", "grasshopper", "cricket", "stick insect",
    "cockroach", "mantis", "cicada", "leafhopper", "lacewing",
    "dragonfly", "damselfly", "red admiral", "ringlet", "monarch butterfly",
    "small white", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin",
    "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster",
    "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig",
    "common sorrel", "zebra", "pig", "wild boar", "warthog",
    "hippopotamus", "ox", "water buffalo", "bison", "ram",
    "bighorn sheep", "Alpine ibex", "hartebeest", "impala", "gazelle",
    "dromedary", "llama", "weasel", "mink", "European polecat",
    "black-footed ferret", "otter", "skunk", "badger", "armadillo",
    "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon",
    "siamang", "guenon", "patas monkey", "baboon", "macaque",
    "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
    "howler monkey", "titi", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur",
    "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda",
    "snoek", "eel", "coho salmon", "rock beauty", "clownfish",
    "sturgeon", "garfish", "lionfish", "pufferfish", "abacus",
    "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier",
    "airliner", "airship", "altar", "ambulance", "amphibious vehicle",
    "analog clock", "apiary", "apron", "waste container", "assault rifle",
    "backpack", "bakery", "balance beam", "balloon", "ballpoint pen",
    "Band-Aid", "banjo", "baluster", "barbell", "barber chair",
    "barbershop", "barn", "barometer", "barrel", "wheelbarrow",
    "baseball", "basketball", "bassinet", "bassoon", "swimming cap",
    "bath towel", "bathtub", "station wagon", "lighthouse", "beaker",
    "military cap", "beer bottle", "beer glass", "bell tower", "baby bib",
    "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse",
    "boathouse", "bobsled", "bolo tie", "poke bonnet", "bookcase",
    "bookstore", "bottle cap", "hunting bow", "bow tie", "brass",
    "bra", "breakwater", "breastplate", "broom", "bucket",
    "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab",
    "cauldron", "candle", "cannon", "canoe", "can opener",
    "cardigan", "car mirror", "carousel", "tool kit", "carton",
    "car wheel", "automated teller machine", "cassette", "cassette player", "castle",
    "catamaran", "CD player", "cello", "mobile phone", "chain",
    "chain-link fence", "chain mail", "chainsaw", "chest", "chiffonier",
    "chime", "china cabinet", "Christmas stocking", "church", "movie theater",
    "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
    "coffee mug", "coffeemaker", "coil", "combination lock", "computer keyboard",
    "confectionery store", "container ship", "convertible", "corkscrew", "cornet",
    "cowboy boot", "cowboy hat", "cradle", "crane (machine)", "crash helmet",
    "crate", "infant bed", "Crock Pot", "croquet ball", "crutch",
    "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone",
    "diaper", "digital clock", "digital watch", "dining table", "dishcloth",
    "dishwasher", "disc brake", "dock", "dog sled", "dome",
    "doormat", "drilling rig", "drum", "drumstick", "dumbbell",
    "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center",
    "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet",
    "fireboat", "fire engine", "fire screen sheet", "flagpole", "flute",
    "folding chair", "football helmet", "forklift", "fountain", "fountain pen",
    "four-poster bed", "freight car", "French horn", "frying pan", "fur coat",
    "garbage truck", "gas mask", "gas pump", "goblet", "go-kart",
    "golf ball", "golf cart", "gondola", "gong", "gown",
    "grand piano", "greenhouse", "grille", "grocery store", "guillotine",
    "barrette", "hair spray", "half-track", "hammer", "hamper",
    "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica",
    "harp", "harvester", "hatchet", "holster", "home theater",
    "honeycomb", "hook", "hoop skirt", "horizontal bar", "horse-drawn vehicle",
    "hourglass", "iPod", "clothes iron", "jack-o'-lantern", "jeans",
    "jeep", "T-shirt", "jigsaw puzzle", "pulled rickshaw", "joystick",
    "kimono", "knee pad", "knot", "lab coat", "ladle",
    "lampshade", "laptop computer", "lawn mower", "lens cap", "paper knife",
    "library", "lifeboat", "lighter", "limousine", "ocean liner",
    "lipstick", "slip-on shoe", "lotion", "speaker", "loupe",
    "sawmill", "magnetic compass", "mail bag", "mailbox", "tights",
    "tank suit", "manhole cover", "maraca", "marimba", "mask",
    "match", "maypole", "maze", "measuring cup", "medicine cabinet",
    "megalith", "microphone", "microwave oven", "military uniform", "milk can",
    "minibus", "miniskirt", "minivan", "missile", "mitten",
    "mixing bowl", "mobile home", "Model T", "modem", "monastery",
    "monitor", "moped", "mortar", "square academic cap", "mosque",
    "mosquito net", "scooter", "mountain bike", "tent", "computer mouse",
    "mousetrap", "moving van", "muzzle", "nail", "neck brace",
    "necklace", "nipple", "notebook computer", "obelisk", "oboe",
    "ocarina", "odometer", "oil filter", "organ", "oscilloscope",
    "overskirt", "bullock cart", "oxygen mask", "packet", "paddle",
    "paddle wheel", "padlock", "paintbrush", "pajamas", "palace",
    "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
    "parking meter", "passenger car", "patio", "payphone", "pedestal",
    "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier",
    "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier",
    "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel",
    "pirate ship", "pitcher", "hand plane", "planetarium", "plastic bag",
    "plate rack", "plow", "plunger", "Polaroid camera", "pole",
    "police van", "poncho", "billiard table", "soda bottle", "pot",
    "potter's wheel", "power drill", "prayer rug", "printer", "prison",
    "projectile", "projector", "hockey puck", "punching bag", "purse",
    "quill", "quilt", "race car", "racket", "radiator",
    "radio", "radio telescope", "rain barrel", "recreational vehicle", "reel",
    "reflex camera", "refrigerator", "remote control", "restaurant", "revolver",
    "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball",
    "ruler", "running shoe", "safe", "safety pin", "salt shaker",
    "sandal", "sarong", "saxophone", "scabbard", "weighing scale",
    "school bus", "schooner", "scoreboard", "CRT screen", "screw",
    "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
    "shoji", "shopping basket", "shopping cart", "shovel", "shower cap",
    "shower curtain", "ski", "ski mask", "sleeping bag", "slide rule",
    "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow",
    "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero",
    "soup bowl", "space bar", "space heater", "space shuttle", "spatula",
    "motorboat", "spider web", "spindle", "sports car", "spotlight",
    "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope",
    "scarf", "stone wall", "stopwatch", "stove", "strainer",
    "tram", "stretcher", "couch", "stupa", "submarine",
    "suit", "sundial", "sunglass", "sunglasses", "sunscreen",
    "suspension bridge", "mop", "sweatshirt", "swimsuit", "swing",
    "switch", "syringe", "table lamp", "tank", "tape player",
    "teapot", "teddy bear", "television", "tennis ball", "thatched roof",
    "front curtain", "thimble", "threshing machine", "throne", "tile roof",
    "toaster", "tobacco shop", "toilet seat", "torch", "totem pole",
    "tow truck", "toy store", "tractor", "semi-trailer truck", "tray",
    "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch",
    "trolleybus", "trombone", "tub", "turnstile", "typewriter keyboard",
    "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase",
    "vault", "velvet", "vending machine", "vestment", "viaduct",
    "violin", "volleyball", "waffle iron", "wall clock", "wallet",
    "wardrobe", "military aircraft", "sink", "washing machine", "water bottle",
    "water jug", "water tower", "whiskey jug", "whistle", "wig",
    "window screen", "window shade", "Windsor tie", "wine bottle", "wing",
    "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck",
    "yawl", "yurt", "website", "comic book", "crossword",
    "traffic sign", "traffic light", "dust jacket", "menu", "plate",
    "guacamole", "consomme", "hot pot", "trifle", "ice cream",
    "ice pop", "baguette", "bagel", "pretzel", "cheeseburger",
    "hot dog", "mashed potato", "cabbage", "broccoli", "cauliflower",
    "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
    "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith",
    "strawberry", "orange", "lemon", "fig", "pineapple",
    "banana", "jackfruit", "custard apple", "pomegranate", "hay",
    "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza",
    "pot pie", "burrito", "red wine", "espresso", "cup",
    "eggnog", "alp", "bubble", "cliff", "coral reef",
    "geyser", "lakeshore", "promontory", "shoal", "seashore",
    "valley", "volcano", "baseball player", "bridegroom", "scuba diver",
    "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
    "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra",
    "stinkhorn mushroom", "earth star", "hen of the woods", "bolete", "ear",
    "toilet paper",
];

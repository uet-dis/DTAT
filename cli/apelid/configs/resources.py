import os


class Resources:
    """Base resources configuration.

    Subclasses should set dataset-specific paths and label collections.
    """

    # Identifier
    resources_name: str = ""

    # Core folders
    DATA_FOLDER: str = ""
    CLEAN_MERGED_DATA_FOLDER: str = ""
    ENCODED_DATA_FOLDER: str = ""
    RAW_PROCESSED_DATA_FOLDER: str = ""

    # Optional/extras
    ORIGINAL_DATA_FOLDER: str | None = None
    ENCODERS_FOLDER: str | None = None
    REPORT_FOLDER: str | None = None
    EMBEDDINGS_FOLDER: str | None = None
    PCA_CACHE_FOLDER: str | None = None

    # Labels
    LABEL_COLUMN: str | None = None
    MAJORITY_LABELS: list[str] = []
    MINORITY_LABELS: list[str] = []


    BALENCED_DATA_FOLDER: str = 'data'

    @staticmethod
    def get_label_name(class_name: str) -> str:
        return class_name.lower().replace(' ', '_').replace('/', '_')

    # ------- Filepath helpers (can be overridden if needed) -------
    @classmethod
    def clean_merged_dir_for_subset(cls, subset: str) -> str:
        if subset == 'full':
            return cls.CLEAN_MERGED_DATA_FOLDER
        return os.path.join(cls.CLEAN_MERGED_DATA_FOLDER, subset)

    @classmethod
    def encoded_dir_for_subset(cls, subset: str) -> str:
        if subset == 'full':
            return cls.ENCODED_DATA_FOLDER
        return os.path.join(cls.ENCODED_DATA_FOLDER, subset)

    @classmethod
    def encoded_filename_for_label(cls, label_safe: str) -> str:
        return f"{cls.resources_name}_{label_safe}_encoded.csv"

    @classmethod
    def list_clean_merged_label_files(cls, input_dir: str, allowed_safe_labels: set[str], subset: str) -> list[str]:
        files: list[str] = []
        if subset == 'full':
            scan_dir = input_dir
            suffix = '_clean_merged.csv'
        else:
            scan_dir = os.path.join(input_dir, subset)
            suffix = f'_{subset}_clean_merged.csv'

        if not os.path.isdir(scan_dir):
            return []

        prefix = f"{cls.resources_name}_"
        for fname in sorted(os.listdir(scan_dir)):
            if not fname.endswith(suffix):
                continue
            if not fname.startswith(prefix):
                continue
            label_safe = fname[len(prefix):-len(suffix)]
            if label_safe in allowed_safe_labels:
                files.append(os.path.join(scan_dir, fname))
        return files

    @classmethod
    def encoded_path_for(cls, subset: str, label_safe: str) -> str:
        base_dir = cls.encoded_dir_for_subset(subset if subset != 'full' else 'full')
        return os.path.join(base_dir, cls.encoded_filename_for_label(label_safe))

    @classmethod
    def clean_merged_path_for(cls, subset: str, label_safe: str, compressed: bool = False) -> str:
        base_dir = cls.clean_merged_dir_for_subset(subset)
        if subset == 'full':
            suffix = '_clean_merged_compressed.csv' if compressed else '_clean_merged.csv'
            fname = f"{cls.resources_name}_{label_safe}{suffix}"
        else:
            suffix = f"_{subset}_clean_merged_compressed.csv" if compressed else f"_{subset}_clean_merged.csv"
            fname = f"{cls.resources_name}_{label_safe}{suffix}"
        return os.path.join(base_dir, fname)



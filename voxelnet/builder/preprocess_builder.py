import voxelnet.core.preprocess as prep

def build_db_preprocess(db_prep_config):
    prepors = []
    for prep_type in db_prep_config.database_preprocessing_step_type:
        if prep_type == 'filter_by_difficulty':
            cfg = db_prep_config.filter_by_difficulty
            prepors.append(prep.DBFilterByDifficulty(cfg.removed_difficulties))
        elif prep_type == 'filter_by_min_num_points':
            cfg = db_prep_config.filter_by_min_num_points
            prepors.append(prep.DBFilterByMinNumPoint({cfg.min_num_point_pairs.key:cfg.min_num_point_pairs.value}))
        else:
            raise ValueError("unknown database prep type")
    return prepors


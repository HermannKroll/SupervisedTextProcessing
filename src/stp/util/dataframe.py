from sklearn.utils import resample

from stp.run_config import RANDOM_STATE


def reduce_dataframe(df, min_class_size, random_state=RANDOM_STATE):
    if len(df) == 0:
        return df
    df = resample(df,
                  replace=False,
                  n_samples=min_class_size,
                  random_state=random_state)
    return df

from multifil.aws.run import manage
from multifil.aws.metas import emit

from multifil.utilities import use_aws
if use_aws:
    from .instance import queue_eater
    from .cluster import watch_cluster

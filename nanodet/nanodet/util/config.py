from .yacs import CfgNode

cfg = CfgNode(new_allowed=True)
cfg.save_dir = "./"
# common params for NETWORK
cfg.model = CfgNode(new_allowed=True)
cfg.model.arch = CfgNode(new_allowed=True)
cfg.model.arch.backbone = CfgNode(new_allowed=True)
cfg.model.arch.fpn = CfgNode(new_allowed=True)
cfg.model.arch.head = CfgNode(new_allowed=True)

# DATASET related params
cfg.data = CfgNode(new_allowed=True)
cfg.data.train = CfgNode(new_allowed=True)
cfg.data.val = CfgNode(new_allowed=True)
cfg.device = CfgNode(new_allowed=True)
cfg.device.precision = 32
# train
cfg.schedule = CfgNode(new_allowed=True)

# logger
cfg.log = CfgNode()
cfg.log.interval = 50

# testing
cfg.test = CfgNode()
# size of images for each device


def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()

    print()
    print("Jia-Baos, cfg.save_dir: ")
    print(cfg.save_dir)

    # common params for NETWORK
    print()
    print("Jia-Baos, cfg.model.weight_averager: ")
    print(cfg.model.weight_averager)
    
    print()
    print("Jia-Baos, cfg.model.arch: ")
    print(cfg.model.arch)

    # DATASET related params
    print()
    print("Jia-Baos, cfg.data.train: ")
    print(cfg.data.train)

    print()
    print("Jia-Baos, cfg.data.val: ")
    print(cfg.data.val)
   
    # device
    print()
    print("Jia-Baos, cfg.device: ")
    print(cfg.device)

    # train
    print()
    print("Jia-Baos, cfg.schedule: ")
    print(cfg.schedule)

    # grad_clip
    print()
    print("Jia-Baos, cfg.grad_clip: ")
    print(cfg.grad_clip)

    # evaluator:
    print()
    print("Jia-Baos, cfg.evaluator: ")
    print(cfg.evaluator)

    # logger
    print()
    print("Jia-Baos, cfg.log: ")
    print(cfg.log)

    # testing
    print()
    print("Jia-Baos, cfg.test: ")
    print(cfg.test)


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "w") as f:
        print("sys.argv[1]: ", sys.argv[1])
        print(cfg, file=f)

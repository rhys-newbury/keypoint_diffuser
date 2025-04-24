from pathlib import Path

import tqdm


out = []
for idx, line in tqdm.tqdm(
    enumerate(open("/home/taco/repos/keypoint_deformer/data/shapenet_split/all.csv")),
    total=51189,
):
    if idx != 0:
        p = Path(
            "/run/user/1000/gvfs/smb-share:server=130.194.128.238,share=slow/Shapenetcore_benchmark"
        )
        p2 = Path(
            "/run/user/1000/gvfs/smb-share:server=130.194.128.238,share=slow/shape-data"
        )
        _, folder, _, name, _ = line.strip().split(",")
        if "test" in line:
            if (p / folder / "points_label" / f"{name}.seg").is_file():
                out.append(line.strip())
        else:
            if (p2 / folder / name).is_dir():
                files = list((p2 / folder / name).rglob("new_samples_*.npy"))
                if len(files) == 5:
                    out.append(line.strip())

    else:
        out.append(line.strip())

update = open("update.csv", "w")
update.write("\n".join(out))
update.close()

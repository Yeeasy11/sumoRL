import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


TURN_DESTINATIONS = {
    "n": {"left": "t_e", "straight": "t_s", "right": "t_w"},
    "e": {"left": "t_s", "straight": "t_w", "right": "t_n"},
    "s": {"left": "t_w", "straight": "t_n", "right": "t_e"},
    "w": {"left": "t_n", "straight": "t_e", "right": "t_s"},
}

ORIGIN_ROUTE_EDGES = {
    "n": "n_t",
    "e": "e_t",
    "s": "s_t",
    "w": "w_t",
}


def _ratio_to_weights(left_ratio: float, straight_ratio: float, right_ratio: float) -> tuple[float, float, float]:
    total = left_ratio + straight_ratio + right_ratio
    if total <= 0:
        raise ValueError("Turn ratios must sum to a positive value")
    return left_ratio / total, straight_ratio / total, right_ratio / total


def write_4way_route(
    out_path: Path,
    begin: int,
    end: int,
    north_vph: int,
    east_vph: int,
    south_vph: int,
    west_vph: int,
    left_ratio: float,
    straight_ratio: float,
    right_ratio: float,
) -> None:
    left_w, straight_w, right_w = _ratio_to_weights(left_ratio, straight_ratio, right_ratio)
    root = ET.Element("routes")

    for origin in ["n", "e", "s", "w"]:
        for turn in ["left", "straight", "right"]:
            destination = TURN_DESTINATIONS[origin][turn]
            route_id = f"route_{origin}_{turn}"
            route_edges = f"{ORIGIN_ROUTE_EDGES[origin]} {destination}"
            ET.SubElement(root, "route", id=route_id, edges=route_edges)

    flow_map = {
        "n": north_vph,
        "e": east_vph,
        "s": south_vph,
        "w": west_vph,
    }
    turn_weights = [("left", left_w), ("straight", straight_w), ("right", right_w)]

    for origin, vph in flow_map.items():
        if vph <= 0:
            continue
        for turn, weight in turn_weights:
            flow_vph = max(1, int(round(vph * weight)))
            ET.SubElement(
                root,
                "flow",
                id=f"flow_{origin}_{turn}",
                route=f"route_{origin}_{turn}",
                begin=str(begin),
                end=str(end),
                vehsPerHour=str(flow_vph),
                departSpeed="max",
                departPos="base",
                departLane="best",
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(out_path, encoding="utf-8", xml_declaration=False)


def main() -> None:
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate a 4-direction single-lane route file with left/straight/right turns.",
    )
    prs.add_argument(
        "--out",
        type=str,
        default="sumo_rl/nets/4way-single-intersection/4way-turns-balanced.rou.xml",
        help="Output route file.",
    )
    prs.add_argument("--begin", type=int, default=0, help="Flow begin time (s)")
    prs.add_argument("--end", type=int, default=3600, help="Flow end time (s)")
    prs.add_argument("--north", type=int, default=600, help="North incoming flow (veh/h)")
    prs.add_argument("--east", type=int, default=600, help="East incoming flow (veh/h)")
    prs.add_argument("--south", type=int, default=600, help="South incoming flow (veh/h)")
    prs.add_argument("--west", type=int, default=600, help="West incoming flow (veh/h)")
    prs.add_argument("--left-ratio", type=float, default=0.2, help="Left-turn share within each incoming flow")
    prs.add_argument("--straight-ratio", type=float, default=0.6, help="Straight share within each incoming flow")
    prs.add_argument("--right-ratio", type=float, default=0.2, help="Right-turn share within each incoming flow")
    args = prs.parse_args()

    write_4way_route(
        out_path=Path(args.out),
        begin=args.begin,
        end=args.end,
        north_vph=args.north,
        east_vph=args.east,
        south_vph=args.south,
        west_vph=args.west,
        left_ratio=args.left_ratio,
        straight_ratio=args.straight_ratio,
        right_ratio=args.right_ratio,
    )
    print(f"Wrote route file: {args.out}")


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def write_single_intersection_route(
    out_path: Path,
    begin: int,
    end: int,
    vehs_per_hour_ns: int,
    vehs_per_hour_we: int,
) -> None:
    """
    Generates a simple flow-based route file for sumo_rl/nets/single-intersection.

    Notes:
    - This net only defines two incoming directions (north->south, west->east).
    - We use vehsPerHour for reproducibility and easier scaling for E4.
    """
    routes = ET.Element("routes")

    ET.SubElement(routes, "route", id="route_ns", edges="n_t t_s")
    ET.SubElement(routes, "route", id="route_we", edges="w_t t_e")

    ET.SubElement(
        routes,
        "flow",
        id="flow_ns",
        route="route_ns",
        begin=str(begin),
        end=str(end),
        vehsPerHour=str(vehs_per_hour_ns),
        departSpeed="max",
        departPos="base",
        departLane="best",
    )
    ET.SubElement(
        routes,
        "flow",
        id="flow_we",
        route="route_we",
        begin=str(begin),
        end=str(end),
        vehsPerHour=str(vehs_per_hour_we),
        departSpeed="max",
        departPos="base",
        departLane="best",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(routes)
    ET.indent(tree, space="    ")
    tree.write(out_path, encoding="utf-8", xml_declaration=False)


def main() -> None:
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate route (.rou.xml) for sumo_rl/nets/single-intersection with vehsPerHour flows.",
    )
    prs.add_argument(
        "--out",
        type=str,
        default="sumo_rl/nets/single-intersection/single-intersection.rou.gen.xml",
        help="Output .rou.xml path",
    )
    prs.add_argument("--begin", type=int, default=0, help="Flow begin time (s)")
    prs.add_argument("--end", type=int, default=600, help="Flow end time (s)")
    prs.add_argument("--ns", type=int, default=600, help="North->South flow (veh/h)")
    prs.add_argument("--we", type=int, default=600, help="West->East flow (veh/h)")
    args = prs.parse_args()

    write_single_intersection_route(
        out_path=Path(args.out),
        begin=args.begin,
        end=args.end,
        vehs_per_hour_ns=args.ns,
        vehs_per_hour_we=args.we,
    )

    print(f"Wrote route file: {args.out}")


if __name__ == "__main__":
    main()


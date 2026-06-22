from __future__ import annotations


class VertexRenderer:
    def draw(self, ax, vertices, plane: str, config) -> None:
        if not config.show_vertices:
            return

        key = "front_vertex_position" if plane == "xz" else "back_vertex_position"
        for vertex in vertices:
            position = vertex.extra_info.get(key)
            if position is None:
                continue
            z = position[2]
            coord = position[0] if plane == "xz" else position[1]
            ax.scatter(
                [z],
                [coord],
                marker="X",
                s=config.vertex_marker_size,
                color="magenta",
                edgecolors="black",
                linewidths=0.8,
                zorder=12,
            )

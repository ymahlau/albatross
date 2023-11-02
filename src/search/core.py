from typing import Optional, Any

from src.search.backup_func import BackupFunc
from src.search.node import Node


def cleanup(root: Optional[Node], exception_node: Optional[Node] = None):
    # delete all nodes using DFS. This is not strictly necessary, but helps the garbage collection
    if root is None:
        return
    stack = [root]
    while stack:
        cur_node = stack.pop()
        if exception_node is not None and cur_node == exception_node:
            continue  # skip exception node for deletion
        if not cur_node.is_leaf():
            for child in cur_node.children.values():
                stack.insert(0, child)
        del cur_node.game
        del cur_node


def update_full_exploration(leaf_list: list[Node]):
    for node in leaf_list:
        # every terminal node is fully explored and fully explored subtrees may have to be propagated upwards
        if node.is_terminal():
            # starting from current node, propagate full exploration upward
            cur_node = node
            while cur_node.is_fully_explored() and not cur_node.is_root():
                # increment fully explored counter of parent
                cur_node.parent.num_children_fully_explored += 1
                cur_node = cur_node.parent


def expand_node_to_depth(
        node: Node,
        max_depth: int,
        discount: float,
        ignore_full_exploration: bool,
) -> tuple[dict[int, list[Node]], list[Node], list[Node]]:
    """
    Expands the node up to specified depth.
    Returns:
        - dictionary of depth -> list of node at this depth
        - list of all new children
        - list of all new leaf nodes
    """
    if node.is_terminal():
        raise ValueError("Cannot expand a terminal node")
    depth_dict = {0: [node]}
    node_list = []
    leaf_list = []
    # expand node using bfs
    queue = [(0, node)]
    while queue:
        cur_depth, cur_node = queue.pop(0)
        # expand node and put children on stack
        cur_node.children = dict()
        for joint_action in cur_node.game.available_joint_actions():
            child = Node(cur_node, joint_action, discount, None, ignore_full_exploration)
            cur_node.children[joint_action] = child
            node_list.append(child)
            # add to queue
            if cur_depth + 1 < max_depth and not child.is_terminal():
                queue.append((cur_depth + 1, child))
            else:
                leaf_list.append(child)
            # add to dict
            if cur_depth + 1 not in depth_dict:
                depth_dict[cur_depth + 1] = []
            depth_dict[cur_depth + 1].append(child)
    return depth_dict, node_list, leaf_list


def backup_depth_dict(
        depth_dict: dict[int, list[Node]],
        max_depth: int,
        backup_func: BackupFunc,
        options: Optional[dict[str, Any]] = None,
) -> None:
    for cur_depth in range(max_depth - 1, -1, -1):
        if cur_depth not in depth_dict:
            continue
        cur_node_list = depth_dict[cur_depth]
        for node in cur_node_list:
            if not node.is_terminal():
                backup_func(node, None, None, options)

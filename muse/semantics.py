# SPDX-License-Identifier: Apache-2.0
"""Operation normalization and semantic history features."""
from __future__ import annotations
import ipaddress
import math
import re
from dataclasses import dataclass, field
import numpy as np

SEMANTIC_STATE_NAMES = (
    "process_spawn_ratio",
    "code_load_ratio",
    "file_open_ratio",
    "file_read_ratio",
    "file_modify_ratio",
    "file_delete_ratio",
    "registry_read_ratio",
    "registry_modify_ratio",
    "network_connect_ratio",
    "network_send_ratio",
    "network_receive_ratio",
    "network_disconnect_ratio",
    "log_event_count",
    "log_unique_files",
    "log_unique_registry_keys",
    "log_unique_remotes",
    "log_child_processes",
    "log_modified_files",
    "log_modified_registry_keys",
    "log_loaded_modules",
    "active_network_sessions",
    "sensitive_resource_seen",
    "system_location_modified",
    "persistence_location_modified",
)


OBJECT_FEATURE_NAMES = (
    "registry",
    "system_path",
    "user_profile",
    "executable_or_library",
    "credential_related",
    "temporary_or_download",
    "external_network",
    "loopback_or_private_network",
    "script",
    "persistence_location",
)


ACTION_GROUP_NAMES = (
    "process_spawn",
    "code_load",
    "file_open",
    "file_read",
    "file_modify",
    "file_delete",
    "registry_read",
    "registry_modify",
    "network_connect",
    "network_send",
    "network_receive",
    "network_disconnect",
)


def action_group(relation: str) -> int:
    mapping = {
        "PROCESS_EXECUTED": 0,
        "MODULE_LOADED": 1,
        "FILE_OPENED": 2,
        "FILE_CLOSED": 2,
        "FILE_READ": 3,
        "FILE_WRITTEN": 4,
        "FILE_CREATED": 4,
        "FILE_TRUNCATED": 4,
        "FILE_RESTORED": 4,
        "FILE_DELETED": 5,
        "REGISTRY_KEY_OPENED": 6,
        "REGISTRY_KEY_LOADED": 6,
        "REGISTRY_VALUE_WRITTEN": 7,
        "REGISTRY_VALUE_DELETED": 7,
        "REGISTRY_KEY_CREATED": 7,
        "REGISTRY_KEY_DELETED": 7,
        "NETFLOW_CREATED": 8,
        "NETFLOW_CONNECTED": 8,
        "NETFLOW_PACKET_SENT": 9,
        "NETFLOW_PACKET_RECEIVED": 10,
        "NETFLOW_DISCONNECTED": 11,
    }
    return mapping.get(relation, 2)


RELATION_MAP = {
    "EVENT_EXECUTE": "PROCESS_EXECUTED",
    "EVENT_FORK": "PROCESS_EXECUTED",
    "EVENT_CLONE": "PROCESS_EXECUTED",
    "EVENT_LOAD": "MODULE_LOADED",
    "EVENT_OPEN": "FILE_OPENED",
    "EVENT_CLOSE": "FILE_CLOSED",
    "EVENT_READ": "FILE_READ",
    "EVENT_WRITE": "FILE_WRITTEN",
    "EVENT_CREATE_OBJECT": "FILE_CREATED",
    "EVENT_UNLINK": "FILE_DELETED",
    "EVENT_DELETE": "FILE_DELETED",
    "EVENT_RENAME": "FILE_WRITTEN",
    "EVENT_TRUNCATE": "FILE_TRUNCATED",
    "EVENT_CONNECT": "NETFLOW_CONNECTED",
    "EVENT_ACCEPT": "NETFLOW_CONNECTED",
    "EVENT_SEND": "NETFLOW_PACKET_SENT",
    "EVENT_SENDTO": "NETFLOW_PACKET_SENT",
    "EVENT_SENDMSG": "NETFLOW_PACKET_SENT",
    "EVENT_RECV": "NETFLOW_PACKET_RECEIVED",
    "EVENT_RECVFROM": "NETFLOW_PACKET_RECEIVED",
    "EVENT_RECVMSG": "NETFLOW_PACKET_RECEIVED",
}


def normalize_relation(relation: str, object_type: str = "") -> str:
    raw = str(relation or "EVENT_OTHER").upper()
    if raw == "EVENT_CLOSE" and object_type == "netflow":
        return "NETFLOW_DISCONNECTED"
    return RELATION_MAP.get(raw, raw)


def _ip_candidates(value: str):
    expanded = value.replace("->", " ").replace(",", " ").replace(";", " ")
    for token in expanded.split():
        token = token.strip("[](){}<>")
        if token.lower() == "netflow":
            continue
        candidates = [token]
        if token.count(":") == 1 and "." in token:
            candidates.append(token.rsplit(":", 1)[0])
        match = re.match(r"^\[([^]]+)\](?::\d+)?$", token)
        if match:
            candidates.append(match.group(1))
        for candidate in candidates:
            try:
                yield ipaddress.ip_address(candidate)
                break
            except ValueError:
                continue


def network_address(label: str):
    """Return the remote-most parseable address from audit labels."""
    addresses = list(_ip_candidates(str(label or "")))
    return addresses[-1] if addresses else None


def object_semantics(label: str, node_type: str) -> tuple[list[float], int]:
    """Extract FreeBSD-aware semantics using the shared ten-slot contract."""
    value = str(label or "").strip().lower()
    path = value.split(" (deleted)", 1)[0]

    registry = False
    system_path = path.startswith((
        "/bin/", "/sbin/", "/usr/bin/", "/usr/sbin/", "/usr/lib/",
        "/lib/", "/libexec/", "/etc/", "/usr/local/etc/", "/boot/", "/var/db/",
    )) or path in {"/bin", "/sbin", "/etc", "/boot"}
    user_profile = path.startswith(("/home/", "/root/", "/usr/home/"))
    executable = path.startswith(("/bin/", "/sbin/", "/usr/bin/", "/usr/sbin/"))
    executable = executable or path.endswith((
        ".so", ".so.0", ".so.1", ".ko", ".elf", ".bin",
    ))
    script = path.endswith((
        ".sh", ".bash", ".csh", ".ksh", ".py", ".pl", ".rb", ".lua",
    ))
    credential = any(token in path for token in (
        "/etc/passwd", "/etc/master.passwd", "/etc/group", "/etc/pwd.db",
        "/etc/spwd.db", "/.ssh/", "authorized_keys", "known_hosts",
        "private_key", "id_rsa", "id_ed25519", "credential", "password",
    ))
    temporary = path.startswith(("/tmp/", "/var/tmp/", "/var/cache/"))
    temporary = temporary or "/.cache/" in path or "/downloads/" in path
    persistence = any(token in path for token in (
        "/etc/crontab", "/var/cron/", "/etc/cron.", "/etc/rc", "/etc/rc.d/",
        "/usr/local/etc/rc.d/", "/etc/periodic/", "authorized_keys",
    ))

    external_network = private_network = False
    if node_type == "netflow":
        address = network_address(value)
        if address is not None:
            private_network = address.is_private or address.is_loopback or address.is_link_local
            external_network = not private_network

    features = [
        registry,
        system_path,
        user_profile,
        executable,
        credential,
        temporary,
        external_network,
        private_network,
        script,
        persistence,
    ]
    if node_type == "netflow":
        category = 8 if external_network else 7
    elif credential:
        category = 5
    elif script:
        category = 4
    elif executable:
        category = 3
    elif system_path:
        category = 2
    elif user_profile:
        category = 1
    else:
        category = 0
    return [float(item) for item in features], category


@dataclass
class ProcessSemanticState:
    counts: np.ndarray = field(
        default_factory=lambda: np.zeros(len(ACTION_GROUP_NAMES), dtype=np.float32)
    )
    event_count: int = 0
    files: set[str] = field(default_factory=set)
    registry_keys: set[str] = field(default_factory=set)
    remotes: set[str] = field(default_factory=set)
    children: set[str] = field(default_factory=set)
    modified_files: set[str] = field(default_factory=set)
    modified_registry: set[str] = field(default_factory=set)
    loaded_modules: set[str] = field(default_factory=set)
    active_sessions: int = 0
    sensitive_seen: bool = False
    system_modified: bool = False
    persistence_modified: bool = False

    def vector(self):
        result = np.zeros(len(SEMANTIC_STATE_NAMES), dtype=np.float32)
        if self.event_count:
            result[: len(ACTION_GROUP_NAMES)] = self.counts / self.event_count
        result[12] = min(1.0, math.log1p(self.event_count) / 14.0)
        result[13] = min(1.0, math.log1p(len(self.files)) / 10.0)
        result[14] = min(1.0, math.log1p(len(self.registry_keys)) / 10.0)
        result[15] = min(1.0, math.log1p(len(self.remotes)) / 10.0)
        result[16] = min(1.0, math.log1p(len(self.children)) / 8.0)
        result[17] = min(1.0, math.log1p(len(self.modified_files)) / 10.0)
        result[18] = min(1.0, math.log1p(len(self.modified_registry)) / 10.0)
        result[19] = min(1.0, math.log1p(len(self.loaded_modules)) / 8.0)
        result[20] = min(1.0, self.active_sessions / 32.0)
        result[21] = float(self.sensitive_seen)
        result[22] = float(self.system_modified)
        result[23] = float(self.persistence_modified)
        return result

    def update(self, relation, object_id, object_type, features):
        group = action_group(relation)
        self.counts[group] += 1.0
        self.event_count += 1
        registry, system, _user, _exe, credential, _temp, _external, _private, _script, persistence = features
        if object_type == "file":
            if registry:
                self.registry_keys.add(object_id)
            else:
                self.files.add(object_id)
        elif object_type == "netflow":
            self.remotes.add(object_id)
        if relation == "PROCESS_EXECUTED":
            self.children.add(object_id)
        if relation == "MODULE_LOADED":
            self.loaded_modules.add(object_id)
        if relation in {"FILE_WRITTEN", "FILE_CREATED", "FILE_TRUNCATED", "FILE_RESTORED", "FILE_DELETED"}:
            self.modified_files.add(object_id)
            self.system_modified = self.system_modified or bool(system)
        if relation in {
            "REGISTRY_VALUE_WRITTEN",
            "REGISTRY_VALUE_DELETED",
            "REGISTRY_KEY_CREATED",
            "REGISTRY_KEY_DELETED",
        }:
            self.modified_registry.add(object_id)
            self.persistence_modified = self.persistence_modified or bool(persistence)
        if relation in {"NETFLOW_CREATED", "NETFLOW_CONNECTED"}:
            self.active_sessions += 1
        elif relation == "NETFLOW_DISCONNECTED":
            self.active_sessions = max(0, self.active_sessions - 1)
        self.sensitive_seen = self.sensitive_seen or bool(credential)

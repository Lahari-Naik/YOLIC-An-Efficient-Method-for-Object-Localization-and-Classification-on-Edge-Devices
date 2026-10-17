"""
Closed-Loop Actuator Logic for Micro-Mobility & Edge Robotics
- Near-field hazard cells (indices 0 to 20): Trigger emergency stop signal (target_speed = 0.0 km/h).
- Mid/Far-field hazard cells (indices 21+): Trigger speed-reduction warning (target_speed = 10.0 km/h).
- Clear road: Normal cruising speed (target_speed = 25.0 km/h).
"""
from typing import Any, Dict, List, Tuple


class ActuatorLogic:
    """
    Evaluates detected active hazard CoIs and determines mechanical vehicle actions.
    """
    def __init__(self, near_field_max_idx: int = 20, normal_speed: float = 25.0, warning_speed: float = 10.0):
        self.near_field_max_idx = near_field_max_idx
        self.normal_speed = normal_speed
        self.warning_speed = warning_speed

    def evaluate(self, active_cois: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates active hazard list.
        Each entry in active_cois: {"coi_id": int, "class": str, "conf": float}
        """
        if not active_cois:
            return {
                "hazard_detected": False,
                "emergency_stop": False,
                "speed_warning": False,
                "target_speed_kmh": self.normal_speed,
                "status_message": "Clear Path - Cruising Speed",
                "severity": "NORMAL"
            }

        # Check for near-field hazard
        near_field_hazards = [h for h in active_cois if h["coi_id"] <= self.near_field_max_idx]
        if near_field_hazards:
            top_hazard = max(near_field_hazards, key=lambda x: x.get("conf", 0.0))
            return {
                "hazard_detected": True,
                "emergency_stop": True,
                "speed_warning": False,
                "target_speed_kmh": 0.0,
                "status_message": f"EMERGENCY BRAKE: Near-field hazard (CoI {top_hazard['coi_id']}: {top_hazard['class']} {top_hazard.get('conf', 0):.2f})",
                "severity": "CRITICAL"
            }

        # Mid/Far-field hazard
        top_hazard = max(active_cois, key=lambda x: x.get("conf", 0.0))
        return {
            "hazard_detected": True,
            "emergency_stop": False,
            "speed_warning": True,
            "target_speed_kmh": self.warning_speed,
            "status_message": f"SPEED WARNING: Ahead hazard detected (CoI {top_hazard['coi_id']}: {top_hazard['class']} {top_hazard.get('conf', 0):.2f})",
            "severity": "WARNING"
        }


def evaluate_actuation(
    active_cois: List[Tuple[int, str, float]],
    near_field_max_idx: int = 20
) -> Dict[str, Any]:
    """
    Helper function supporting active_cois as tuples (coi_id, class_name, confidence).
    """
    logic = ActuatorLogic(near_field_max_idx=near_field_max_idx)
    formatted = [{"coi_id": h[0], "class": h[1], "conf": h[2]} for h in active_cois]
    return logic.evaluate(formatted)

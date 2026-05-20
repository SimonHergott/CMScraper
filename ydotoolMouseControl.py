import subprocess
import time


# Uses external controller to work on wayland
class YdotoolMouseController:
    def move(self, x, y):
        """Déplacement absolu."""
        subprocess.run(["ydotool", "mousemove", "-a", str(int(x)), str(int(y))], check=False)

    def click(self, button="left"):
        """Clic (0xC0 = gauche, 0xC1 = droit, 0xC2 = milieu)."""
        btn_map = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
        code = btn_map.get(button, "0xC0")
        subprocess.run(["ydotool", "click", code], check=False)

    def scroll(self, amount):
        """Scroll vertical. Négatif = bas."""
        # Note: la syntaxe de scroll dépend de la version de ydotool, 
        # wheel utilise -1 / 1 (ou des valeurs multiples).
        direction = "-1" if amount < 0 else "1"
        steps = abs(amount) // 100 or 1
        for _ in range(steps):
            subprocess.run(["ydotool", "wheel", direction], check=False)
            time.sleep(0.05)
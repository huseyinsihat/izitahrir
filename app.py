"""Hugging Face Space girişi. Arayüz ve API aynı adreste açılır."""

try:
    import spaces
except ImportError:  # yerel kurulumda ZeroGPU paketi yoktur
    spaces = None

if spaces is not None:

    @spaces.GPU(duration=60)
    def gpu_ready() -> str:
        return "hazir"


from app.serve import main as serve_main


def main() -> None:
    if spaces is not None:
        gpu_ready()
    serve_main()


if __name__ == "__main__":
    main()

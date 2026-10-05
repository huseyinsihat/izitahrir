"""Hugging Face Space girişi. Arayüz ve API aynı adreste açılır."""

try:
    import spaces
except ImportError:  # yerel kurulumda ZeroGPU paketi yoktur
    spaces = None

if spaces is not None:

    @spaces.GPU(duration=60)
    def gpu_ready() -> str:
        return "hazir"


from app.serve import main

if __name__ == "__main__":
    main()

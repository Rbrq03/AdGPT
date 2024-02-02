import argparse
from metrics.GSS import GSS

def get_args():
    parser = argparse.ArgumentParser(description='AdGPT')
    parser.add_argument('--GSS', action='store_true', help='use GSS')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    if args.GSS:
        prompt = """
        This advertisement from the WWF leverages the image of a turtle in the ocean, accompanied by a digital "Save" button, to highlight the urgent need for environmental conservation, particularly marine life protection. By creating an emotional appeal through wildlife imagery and presenting a clear call to action, the ad encourages viewers to contribute to saving the planet, either through donations or engagement in conservation efforts. The use of the "Don't Save" versus "Save" buttons cleverly emphasizes the impact of individual choices, urging immediate action to support the cause and make a positive difference for living creatures and their habitats.
        """
        image_path = "./img/turtle.png"
        SD_model_ID = "stabilityai/sdxl-turbo"
        clip_model_ID = "openai/clip-vit-base-patch32"
        num_inference_steps = 4
        devices = "cuda:0"
        rater = GSS(SD_model_ID, clip_model_ID, num_inference_steps, devices)
        result = rater.get_score(prompt, image_path)
        print(result)
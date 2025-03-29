import imageio


def rendering(env,model):
    frames = []
    obs, _ = env.reset()

    for _ in range(200):
        frame = env.render()  # Now returns an image with obstacles & objects
        frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(int(action))

        if done:
            break

    # Ensure frames have correct shape (height, width, channels)
    frames = [frame[:, :, :3] for frame in frames]  # Remove alpha channel if present

    # Save as a video
    imageio.mimsave('auv_simulation.mp4', frames, fps=10)
    print("Simulation Video Saved!")





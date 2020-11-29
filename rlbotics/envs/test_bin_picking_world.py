from rlbotics.envs.gym.bin_picking_gym import BinPickingGym
import time

def main():
    env = BinPickingGym('kuka', render=True)
    env.reset()
    time.sleep(10)

if __name__ == '__main__':
    main()
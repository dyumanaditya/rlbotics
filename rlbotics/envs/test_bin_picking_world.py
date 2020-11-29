from rlbotics.envs.worlds.bin_picking_world import BinPickingWorld
import time

def main():
    env = BinPickingWorld('kuka', render=True)
    env.reset()
    time.sleep(10)

if __name__ == '__main__':
    main()
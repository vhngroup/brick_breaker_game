import cv2 as cv
import mediapipe as mp
import pygame
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
pygame.init()

WIDTH, HEIGTH = 800, 600
scren = pygame.display.set_mode((WIDTH, HEIGTH))
pygame.display.set_caption("Game of Brick Breaker")
clock = pygame.time.Clock()
running = True

paddle_image = pygame.image.load("./statics/bar_cloud_fly.png")
paddle_image = pygame.transform.scale(paddle_image, (150, 40))
paddle_image_x, paddle_image_y = WIDTH//2, HEIGTH-100
paddle_width, paddle_height = 150, 40
paddle_speed = 10

ball_image = pygame.image.load("./statics/ball.png")
ball_image = pygame.transform.scale(ball_image, (40, 40))
ball_image_x, ball_image_y = WIDTH//2, HEIGTH//2
ball_speed_x, ball_speed_y = 8,8
ball_radius = 20

block_image = pygame.image.load("./statics/block.png")
block_rows, block_columns = 4,8
block_width, block_height = WIDTH // block_columns, 50
block_image = pygame.transform.scale(block_image, (block_width, block_height))
blocks=[]

#create a blocks structure
for row in range(block_rows):
    for column in range(block_columns):
        blocks.append(pygame.Rect(column * block_width, row * block_height, block_width, block_height))

cap = cv.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5, max_num_hands = 1) as hands:
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        #flip the frame with the x axis
        frame = cv.flip(frame, 1)
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger = hand_landmarks.landmark[8]
                paddle_image_x = int(index_finger.x * WIDTH) - paddle_width//2
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        ball_image_x += ball_speed_x
        ball_image_y += ball_speed_y

        #Extreme wall collision
        if ball_image_x - ball_radius < 0 or ball_image_x + ball_radius > WIDTH:
            ball_speed_x *= -1

        if ball_image_y - ball_radius < 0:
            ball_speed_y *= -1
        
        if ball_image_y + ball_radius > HEIGTH:
            running = False
        
        #paddle collision
        if paddle_image_x  < 0:
            paddle_image_x = 0
        if paddle_image_x + paddle_width > WIDTH:
            paddle_image_x = WIDTH - paddle_width
        
        if paddle_image_x < ball_image_x < paddle_image_x + paddle_width and ball_image_y + ball_radius >= paddle_image_y:
            ball_speed_y *= -1

        for block in blocks:
            if block.collidepoint(ball_image_x, ball_image_y):
                blocks.remove(block)
                ball_speed_y *= -1
                break

        # flip the frame with the x axis
        rgb_frame = cv.flip(image, 1)
        #resize the frame
        rgb_frame = cv.resize(rgb_frame, (WIDTH, HEIGTH))
        #rotate the matricial frame
        rgb_frame = np.rot90(rgb_frame)
        #camera frame is a background
        rgb_frame = pygame.surfarray.make_surface(rgb_frame)


        #update the screen and coordinates
        scren.blit(rgb_frame, (0, 0))

        #render the paddle
        scren.blit(paddle_image, (paddle_image_x, paddle_image_y))

        #render the ball
        scren.blit(ball_image, (ball_image_x, ball_image_y))

        #render the blocks
        for block in blocks:
            scren.blit(block_image, (block.x - ball_radius, block.y - ball_radius))

        pygame.display.flip()
        clock.tick(30)


        if cv.waitKey(1) & 0xFF == 27:
            running = False
            break

cap.release()
cv.destroyAllWindows()
# Product Vision

## Big Picture

**B** lets a coach, analyst, broadcaster, or player review previous football
footage and ask: "What if this movement had happened instead?"

The user draws arrows for potential player runs, passes, presses, drops, or other
tactical situations. The system then resimulates the game from that moment using
trained tracking data in our database. The important behavior is collective:
every player and the ball should react, move, and interact realistically based on
the new tactical intent.

The output should teach movement visually. The audience should understand how to
move or react because they can see themselves, their teammates, and opponents
doing it in the simulation.

## First Checkpoint

The first checkpoint is a **2D bird's-eye soccer field simulation**.

It is intentionally not a hyper-realistic video. The 2D checkpoint must prove the
hardest part first:

1. Load a previous match moment.
2. Draw arrows for potential movements or situations.
3. Generate a full-game reaction from that moment.
4. Show actual vs. alternative on a 2D pitch.
5. Make the alternative movement understandable to a coach, analyst, player, or
   broadcast viewer.

If this does not work in 2D, a 3D video layer will only make the wrong behavior
more expensive to render. If it works in 2D, the same trajectory output can later
power a photorealistic video layer.

## Product Principle

The user controls intent, not animation.

An arrow is a tactical constraint. The model is responsible for the rest of the
game: spacing, reactions, movement timing, pressure, ball behavior, and realistic
interactions learned from game data.


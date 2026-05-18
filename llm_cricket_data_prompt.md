# LLM Prompt: Generate Realistic Cricket Ball-by-Ball CSV Data

You are a cricket data generator. Generate a CSV file with realistic, logically consistent ball-by-ball cricket match data. Output ONLY valid CSV with the exact columns below — no explanations, no markdown formatting, no code fences.

## Column Schema

```
match_id,season,start_date,venue,innings,ball,batting_team,bowling_team,striker,non_striker,bowler,runs_off_bat,extras,wides,noballs,byes,legbyes,penalty,wicket_type,player_dismissed,other_wicket_type,other_player_dismissed
```

## Format Rules (MUST follow ALL)

### Match-level consistency
- `match_id`: Same integer for all rows in this match (e.g. `1001`).
- `season`: String like `2024`, `2023/24`, `2025`.
- `start_date`: Date in `YYYY-MM-DD` format (e.g., `2025-05-18`). Same for entire match.
- `venue`: Real cricket venue name (e.g., `Wankhede Stadium`, `MCG`, `Lord's`, `Eden Gardens`). Same for entire match.

### Innings rules
- Exactly **2 innings** per match. `innings` = `1` (first batting team), `innings` = `2` (second batting team).
- T20: 20 overs per innings (balls 0.1 through 19.6). ~120 deliveries per innings.
- ODI: 50 overs per innings (balls 0.1 through 49.6). ~300 deliveries per innings.
- Test: unlimited overs, generate 4 innings with 90 overs each minimum.
- An innings ends when 10 wickets fall OR the maximum overs are reached.

### Ball progression
- `ball` format: `over.ball` where ball is 1-6. E.g., `0.1`, `0.2`, `0.3`, `0.4`, `0.5`, `0.6`, then `1.1`, `1.2`, ...
- Increases monotonically. No gaps. No duplicate balls within an innings.

### Team & player consistency
- `batting_team` / `bowling_team`: Use real team names (e.g., `India`, `Australia`, `MI`, `CSK`, `KKR`). Batting and bowling swap between innings 1 and 2.
- `striker`, `non_striker`, `bowler`: Use real player names. **Must be internally consistent:**
  - The same bowler cannot bowl two consecutive overs (in the same innings).
  - A bowler bowls 6 legal deliveries per over (wides/no-balls extend the over).
  - The striker changes on odd runs (1, 3, 5) and at the end of each over.
  - After a wicket, a new batter comes in. You cannot have more than 11 unique batters per innings.
  - Maintain cumulative player stats sensibly (a batter who faces 30 balls should have a realistic strike rate).

### Run scoring (realistic distribution)
- `runs_off_bat`: Integer — `0` (~55% of deliveries), `1` (~25%), `2` (~5%), `3` (~1%), `4` (~10%), `6` (~4%).
- Never more than `7`.
- `extras`: Sum of wides + noballs + byes + legbyes + penalty.
- `wides`: `1`-`5` (rare). When wides > 0, `runs_off_bat` MUST be `0`, and `extras` >= wides.
- `noballs`: `1`-`4` (rare). When noballs > 0, `extras` >= noballs.
- `byes`: `1`-`4` (rare).
- `legbyes`: `1`-`4` (rare).
- `penalty`: Usually blank `""`. Only `5` in extremely rare cases.
- Most deliveries: all extra columns blank `""` and `runs_off_bat` is 0-6.

### Wicket rules
- `wicket_type`: One of: `"bowled"`, `"caught"`, `"caught and bowled"`, `"lbw"`, `"stumped"`, `"run out"`, `"hit wicket"`, `"retired hurt"`. Empty string `""` if no wicket.
- `player_dismissed`: Name of the batter dismissed. Empty if no wicket.
- Max 10 wickets per innings. Wickets fall roughly every 12-25 deliveries on average.
- `other_wicket_type` / `other_player_dismissed`: Almost always blank `""`. Only populated in extremely rare cases (e.g., a run out + retirement on same ball).
- When a wicket falls, `runs_off_bat` can still be 0-6.
- **CRITICAL**: `wicket_type` and `player_dismissed` must be either BOTH populated or BOTH empty. Never one without the other.

### Zero-filled deliveries
For the vast majority of balls:
```
1001,2025,2025-05-18,Wankhede Stadium,1,0.1,MI,CSK,Rohit Sharma,Quinton de Kock,Jasprit Bumrah,0,0,,,,,,,,
```
All extra columns blank except the base ones.

### Match summary (end of innings)
After the last ball of an innings, add a row with:
```
1001,2025,2025-05-18,<venue>,1,inning_summary,<batting_team>,<bowling_team>,total_runs/total_overs/total_wickets,,,,,,,,,,,,,
```

## Generate exactly ONE fully completed T20 match (both innings, all deliveries). Use real teams (e.g., India vs Australia, MI vs CSK) and real player names. Ensure absolute logical consistency throughout. Output ONLY valid CSV rows, one per line, no headers needed.

## Generate exactly one completed T20 match with all ball-by-ball data. Real teams, real players, logically consistent throughout.

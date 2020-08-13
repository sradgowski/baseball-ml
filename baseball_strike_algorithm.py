import seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from decimal import Decimal

import numpy as np
import matplotlib.pyplot as plt

def make_meshgrid(ax, h=.02):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def draw_boundary(ax, clf):
    xx, yy = make_meshgrid(ax)
    return plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.5)

aaron_judge = pd.read_csv('aaron_judge.csv')
david_ortiz = pd.read_csv('david_ortiz.csv')
jose_altuve = pd.read_csv('jose_altuve.csv')
tenths = [x * Decimal('0.1') for x in range(1, 51)]
optimal_gamma = 0
optimal_C = 0


def ask_user():
  answer = input("""
  Which player would you like to analyze?:
  A: Aaron Judge
  B: David Ortiz
  C: Jose Altuve
  """)
  if (answer == 'A') or (answer == 'a'):
    return aaron_judge
  if (answer == 'B') or (answer == 'b'):
    return david_ortiz
  if (answer == 'C') or (answer == 'c'):
    return jose_altuve
  else:
    print("Invalid input. Please try again.")
    return None

def graph_strikezone():
  player = ask_user()
  fig, ax = plt.subplots()
  plt.title("{}'s Strike Zone".format(player.player_name[1]))
  player['type'] = player['type'].map({'S':1, 'B':0})
  player = player.dropna(subset = ['plate_x', 'plate_z', 'type'])
  strikes = player[player['type']==1.0]
  balls = player[player['type']==0.0]

  labels = ["Strike", "Ball"]
  plt.scatter(strikes['plate_x'], strikes['plate_z'], c='red', cmap=plt.cm.coolwarm, alpha=0.25)
  plt.scatter(balls['plate_x'], balls['plate_z'], c='blue', cmap = plt.cm.coolwarm, alpha=0.25)
  plt.legend(labels)

  training_set, validation_set = train_test_split(player, random_state = 1)
  score = 0.80
  for g in range(1, 5):
    for d in tenths:
      classifier = SVC(kernel='rbf', gamma = g, C = d)
      classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
      current_score = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])
      if current_score > score:
        score = current_score
        optimal_gamma = g
        optimal_C = d

  print("Gamma: {}".format(optimal_gamma))
  print("C: {}".format(optimal_C))

  classifier = SVC(kernel='rbf', gamma = optimal_gamma, C = optimal_C)
  classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
  draw_boundary(ax, classifier)
  ax.set_ylim(-2, 6)
  ax.set_xlim(-3, 3)
  plt.xlabel('X Position of Pitch')
  plt.ylabel('Z Position of Pitch')
  plt.show()
  final_score = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])

  print("Optimal accuracy score of model: {}".format(round(final_score, 4)))

graph_strikezone()
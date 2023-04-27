import numpy as np


def angleDegreesToRadians(angleDegrees):

    return [angle * np.pi / 180 for angle in angleDegrees]

def angleRadiansToDegrees(angleRadians):

    return [angle * 180 / np.pi for angle in angleRadians]

# Fix hack used in viewer to get angles in correct coordinate system
def unhackAngles(angles):

    return [-angles[1], angles[0]]

def vectorToAngles(point1, point2):
    
    # Figure out what to do here....

    return [angle1, angle2]

# Transform point by rigid transformation
def transformPoint(point, rigidTransformation):

    augmentedPoint = [
        [point[0]],
        [point[1]],
        [point[2]],
        [1]]
    transformedPoints = rigidTransformation @ augmentedPoint

    return [point for point in transformedPoints.flatten()[:-1]]

# Transform angles by rigid transformation
def rotateAngle(angle, rotationMatrix):

    #print(angle)
    #print(rotationMatrix)

    # Create vector from angle
    angleVector = [np.tan(angle[1]), np.tan(angle[0]), 1]

    # Augment and perform 
    rotatedVector = rotationMatrix @ np.array(angleVector)[:,None]

    # Find angles from vector
    alpha = np.arctan2(rotatedVector[1,0], rotatedVector[2,0])
    beta = np.arctan2(rotatedVector[0,0], rotatedVector[2,0])
    
    return [alpha, beta]

# Compute intersection of tool in pedicle plane
def findToolIntersection(screwAngle, screwPoint, toolAngle, toolPoint):

    # Compute pedicle screw plane
    planeVector = np.array([np.tan(screwAngle[1]), np.tan(screwAngle[0]), 1])
    planePoint = np.array(screwPoint)
    planeIntercept = planeVector[:,None].T @ planePoint[:,None]

    # Formulate linear algebra solution to intersection of screw plane and trajectory line
    intersectionMatrix = [
            [1, 0, 0, -np.tan(toolAngle[1])],
            [0, 1, 0, -np.tan(toolAngle[0])],
            [0, 0, 1, -1],
            [planeVector[0], planeVector[1], planeVector[2], 0]]
    intersectionVector = np.array([
            [toolPoint[0]],
            [toolPoint[1]],
            [toolPoint[2]],
            [planeIntercept[0,0]]])
    intersectionSolution = np.linalg.inv(intersectionMatrix) @ intersectionVector

    return [point for point in intersectionSolution[:,0]][:-1]

def computeAngularError(angle1, angle2):

    return np.linalg.norm(np.array(angle1) - np.array(angle2))

def pointDistance(point1, point2):

    return np.linalg.norm(point1 - point2)

def main():
# Input information 
    navigationPoint_mm = list(map(float, input("Navigation point: ").split()))
    navigationAngles_degrees = list(map(float, input("Navigation angles: ").split()))
    screwPediclePoint_mm = list(map(float, input("Screw pedicle point: ").split()))
    screwAxisPoint_mm = list(map(float, input("Screw axis point: ").split()))

    # Process input data
    navigationAngles_radians = angleDegreesToRadians(navigationAngles_degrees)
    screwAngle_radians = vectorToAngles(screwPediclePoint_mm, screwAxisPoint_mm)
    screwAngle_degrees = angleRadiansToDegrees(screwAngle_radians)

    # Compute tool intersection
    navigationIntersectionPoint_mm = findToolIntersection(screwAngle_radians , screwPediclePoint_mm, navigationAngles_radians, navigationPoint_mm)
    print("Tool pedicle plane intersection:", navigationIntersectionPoint_mm)

    # Compute positional error
    positionalError_mm = pointDistance(navigationIntersectionPoint_mm, screwPediclePoint_mm)
    print("Tool positional error in pedicle plane:", positionalError_mm)

    # Compute angular error
    angularError_degrees = computeAngularError(navigationAngles_degrees, screwAngle_degrees)
    print("Tool anglular error against screw:", angularError_degrees)

if __name__ == "__main__":
    main()

from devito import SubDomain


class LeftY3D(SubDomain):
    name = 'left_y'
    model_spacing = (0, 0, 0)
    nbl = 0
    position = 0

    def set_parameters(self, model_spacing, nbl, position):
        self.model_spacing=model_spacing
        self.nbl=nbl
        self.position=position

    def define(self, dimensions):
        x, y, z = dimensions
        nbl, spacing = self.nbl, self.model_spacing
        dx, dy, dz = spacing
        position=nbl+int(self.position/dy)
        return {x: x, z: z, y: ('left', position)}


class RightY3D(SubDomain):
    name = 'right_y'
    model_spacing = (0, 0, 0)
    nbl = 0
    model_shape = (0, 0, 0)
    position = 0

    def set_parameters(self, model_spacing, nbl, model_shape, position):
        self.model_spacing=model_spacing
        self.nbl=nbl
        self.model_shape=model_shape
        self.position=position

    def define(self, dimensions):
        x, y, z = dimensions
        nbl, spacing, shape = self.nbl, self.model_spacing, self.model_shape
        dx, dy, dz = spacing
        nx, ny, nz = shape
        position=ny-int(self.position/dy)+nbl
        return {x: x, z: z, y: ('right', position)}


class LeftX3D(SubDomain):
    name = 'left_x'
    model_spacing = (0, 0, 0)
    nbl = 0
    position = 0

    def set_parameters(self, model_spacing, nbl, position):
        self.model_spacing = model_spacing
        self.nbl = nbl
        self.position = position

    def define(self, dimensions):
        x, y, z = dimensions
        nbl, spacing = self.nbl, self.model_spacing
        dx, dy, dz = spacing
        position = nbl + int(self.position / dx)
        return {y: y, z: z, x: ('left', position)}


class RightX3D(SubDomain):
    name = 'right_x'
    model_spacing = (0, 0)
    nbl = 0
    model_shape = (0, 0, 0)
    position = 0

    def set_parameters(self, model_spacing, nbl, model_shape, position):
        self.model_spacing = model_spacing
        self.nbl = nbl
        self.model_shape = model_shape
        self.position = position

    def define(self, dimensions):
        x, y, z = dimensions
        nbl, spacing, shape = self.nbl, self.model_spacing, self.model_shape
        dx, dy, dz = spacing
        nx, ny, nz = shape
        position = nx - int(self.position / dx) + nbl
        return {y: y, z: z, x: ('right', position)}


class LeftZ3D(SubDomain):
    name = 'left_z'
    model_spacing = (0, 0, 0)
    nbl = 0
    position = 0

    def set_parameters(self, model_spacing, nbl, position):
        self.model_spacing = model_spacing
        self.nbl = nbl
        self.position = position

    def define(self, dimensions):
        x, y, z = dimensions
        nbl, spacing = self.nbl, self.model_spacing
        dx, dy, dz = spacing
        position = nbl + int(self.position / dz)
        return {x: x, y: y, z: ('left', position)}


class RightZ3D(SubDomain):
    name = 'right_z'
    model_spacing = (0, 0, 0)
    nbl = 0
    model_shape=(0, 0, 0)
    position = 0

    def set_parameters(self, model_spacing, nbl, model_shape, position):
        self.model_spacing=model_spacing
        self.nbl=nbl
        self.model_shape=model_shape
        self.position=position

    def define(self, dimensions):
        x, y, z = dimensions
        nbl, spacing, shape = self.nbl, self.model_spacing, self.model_shape
        dx, dy, dz = spacing
        nx, ny, nz = shape
        position = nz - int(self.position / dz) + nbl
        return {x: x, y: y, z: ('right', position)}


class LeftY2D(SubDomain):
    name = 'left_y'
    model_spacing=(0,0)
    nbl=0
    position = 0

    def set_parameters(self, model_spacing, nbl, position):
        self.model_spacing=model_spacing
        self.nbl=nbl
        self.position=position

    def define(self, dimensions):
        x, y = dimensions
        nbl, spacing = self.nbl, self.model_spacing
        dx, dy = spacing
        position=nbl+int(self.position/dy)
        return {x: x, y: ('left', position)}


class RightY2D(SubDomain):
    name = 'right_y'
    model_spacing = (0, 0)
    nbl = 0
    model_shape=(0,0)
    position = 0

    def set_parameters(self, model_spacing, nbl, model_shape, position):
        self.model_spacing=model_spacing
        self.nbl=nbl
        self.model_shape=model_shape
        self.position=position

    def define(self, dimensions):
        x, y = dimensions
        nbl, spacing, shape = self.nbl, self.model_spacing, self.model_shape
        dx, dy = spacing
        nx, ny = shape
        position=ny-int(self.position/dy)+nbl
        # print(position, shape)
        return {x: x, y: ('right', position)}


class LeftX2D(SubDomain):
    name = 'left_x'
    model_spacing=(0,0)
    nbl=0
    position = 0

    def set_parameters(self, model_spacing, nbl, position):
        self.model_spacing=model_spacing
        self.nbl=nbl
        self.position=position

    def define(self, dimensions):
        x, y = dimensions
        nbl, spacing = self.nbl, self.model_spacing
        dx, dy = spacing
        position=nbl+int(self.position/dx)
        return {y: y, x: ('left', position)}


class RightX2D(SubDomain):
    name = 'right_x'
    model_spacing = (0, 0)
    nbl = 0
    model_shape=(0,0)
    position = 0

    def set_parameters(self, model_spacing, nbl, model_shape, position):
        self.model_spacing=model_spacing
        self.nbl=nbl
        self.model_shape=model_shape
        self.position=position

    def define(self, dimensions):
        x, y = dimensions
        nbl, spacing, shape = self.nbl, self.model_spacing, self.model_shape
        dx, dy = spacing
        nx, ny = shape
        position=nx-int(self.position/dx)+nbl
        print(position, shape)
        return {y: y, x: ('right', position)}

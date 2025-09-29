import React, { useState } from 'react'
import { observer } from 'mobx-react-lite'
import { useStores } from '../../stores/context'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, Modal, ConfirmModal } from '../ui'

export const ConfigProfiles = observer(() => {
  const { configStore } = useStores()
  const [loading, setLoading] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})

  // Modal states
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [selectedProfileForDelete, setSelectedProfileForDelete] = useState<string>('')

  // Form states
  const [newProfileName, setNewProfileName] = useState('')
  const [profileDescription, setProfileDescription] = useState('')

  const handleCreateProfile = async () => {
    if (!newProfileName.trim()) {
      setErrors({ profileName: 'Profile name is required' })
      return
    }

    if (configStore.availableProfiles.includes(newProfileName.trim())) {
      setErrors({ profileName: 'A profile with this name already exists' })
      return
    }

    setLoading(true)
    setErrors({})

    try {
      const result = await configStore.saveProfile(newProfileName.trim())

      if (result.success) {
        setNewProfileName('')
        setProfileDescription('')
        setShowCreateModal(false)
      } else {
        setErrors({ profileName: result.error || 'Failed to create profile' })
      }
    } catch (error) {
      console.error('Failed to create profile:', error)
      setErrors({ profileName: 'Failed to create profile. Please try again.' })
    } finally {
      setLoading(false)
    }
  }

  const handleLoadProfile = async (profileName: string) => {
    setLoading(true)
    setErrors({})

    try {
      const result = await configStore.loadProfile(profileName)

      if (!result.success) {
        setErrors({ load: result.error || 'Failed to load profile' })
      }
    } catch (error) {
      console.error('Failed to load profile:', error)
      setErrors({ load: 'Failed to load profile. Please try again.' })
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteProfile = async () => {
    if (!selectedProfileForDelete) return

    setLoading(true)
    setErrors({})

    try {
      const result = await configStore.deleteProfile(selectedProfileForDelete)

      if (result.success) {
        setSelectedProfileForDelete('')
        setShowDeleteModal(false)
      } else {
        setErrors({ delete: result.error || 'Failed to delete profile' })
      }
    } catch (error) {
      console.error('Failed to delete profile:', error)
      setErrors({ delete: 'Failed to delete profile. Please try again.' })
    } finally {
      setLoading(false)
    }
  }

  const openDeleteModal = (profileName: string) => {
    setSelectedProfileForDelete(profileName)
    setShowDeleteModal(true)
  }

  const getProfileDescription = (profileName: string) => {
    switch (profileName) {
      case 'default':
        return 'Standard configuration with balanced settings for most users'
      case 'performance':
        return 'Optimized for performance with lower quality settings for faster processing'
      case 'quality':
        return 'High-quality settings for best detection accuracy and visual quality'
      default:
        return 'Custom user-created profile'
    }
  }

  const getProfileIcon = (profileName: string) => {
    switch (profileName) {
      case 'default':
        return (
          <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
      case 'performance':
        return (
          <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        )
      case 'quality':
        return (
          <svg className="w-5 h-5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
          </svg>
        )
      default:
        return (
          <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
        )
    }
  }

  return (
    <div className="space-y-6">
      {(errors.load || errors.delete) && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-800">{errors.load || errors.delete}</p>
            </div>
          </div>
        </div>
      )}

      {/* Active Profile */}
      <Card>
        <CardHeader>
          <CardTitle>Current Profile</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between p-4 bg-blue-50 dark:bg-blue-900 rounded-lg">
            <div className="flex items-center space-x-3">
              {getProfileIcon(configStore.currentProfile)}
              <div>
                <h3 className="font-medium text-gray-900 dark:text-white">
                  {configStore.currentProfile}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {getProfileDescription(configStore.currentProfile)}
                </p>
              </div>
            </div>
            {configStore.hasUnsavedChanges && (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                Unsaved Changes
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Profile Management */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Configuration Profiles</CardTitle>
            <Button
              onClick={() => setShowCreateModal(true)}
              size="sm"
            >
              Create New Profile
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {configStore.availableProfiles.map((profileName) => (
              <div
                key={profileName}
                className={`flex items-center justify-between p-4 border rounded-lg transition-colors ${
                  profileName === configStore.currentProfile
                    ? 'border-blue-300 bg-blue-50 dark:bg-blue-900 dark:border-blue-700'
                    : 'border-gray-200 hover:border-gray-300 dark:border-gray-700 dark:hover:border-gray-600'
                }`}
              >
                <div className="flex items-center space-x-3">
                  {getProfileIcon(profileName)}
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white">
                      {profileName}
                      {profileName === configStore.currentProfile && (
                        <span className="ml-2 text-xs text-blue-600 dark:text-blue-400">
                          (Active)
                        </span>
                      )}
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {getProfileDescription(profileName)}
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  {profileName !== configStore.currentProfile && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleLoadProfile(profileName)}
                      disabled={loading}
                    >
                      Load
                    </Button>
                  )}

                  {profileName !== 'default' && (
                    <Button
                      size="sm"
                      variant="danger"
                      onClick={() => openDeleteModal(profileName)}
                      disabled={loading || profileName === configStore.currentProfile}
                    >
                      Delete
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Profile Statistics */}
      <Card>
        <CardHeader>
          <CardTitle>Profile Statistics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {configStore.availableProfiles.length}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Total Profiles
              </div>
            </div>

            <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {configStore.hasUnsavedChanges ? '1' : '0'}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Unsaved Changes
              </div>
            </div>

            <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <div className={`text-2xl font-bold ${configStore.isValid ? 'text-green-600' : 'text-red-600'}`}>
                {configStore.isValid ? 'Valid' : 'Invalid'}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Configuration Status
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Create Profile Modal */}
      <Modal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Create New Profile"
      >
        <div className="space-y-4">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Create a new configuration profile using your current settings.
          </p>

          <Input
            label="Profile Name"
            value={newProfileName}
            onChange={(e) => setNewProfileName(e.target.value)}
            error={errors.profileName}
            placeholder="Enter profile name"
            fullWidth
          />

          <div className="flex justify-end space-x-3">
            <Button
              variant="outline"
              onClick={() => {
                setShowCreateModal(false)
                setNewProfileName('')
                setErrors({})
              }}
              disabled={loading}
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreateProfile}
              loading={loading}
              disabled={loading || !newProfileName.trim()}
            >
              Create Profile
            </Button>
          </div>
        </div>
      </Modal>

      {/* Delete Confirmation Modal */}
      <ConfirmModal
        isOpen={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        onConfirm={handleDeleteProfile}
        title="Delete Profile"
        message={`Are you sure you want to delete the "${selectedProfileForDelete}" profile? This action cannot be undone.`}
        confirmText="Delete"
        confirmVariant="danger"
        loading={loading}
      />
    </div>
  )
})
